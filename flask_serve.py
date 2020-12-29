import datetime
import os
import random
import re
import math
import multiprocessing
import subprocess
import sqlite3
import time

import gmpy2
from flask import Flask, Response, g, render_template, request
from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length
from wtforms_components import IntegerField, DateField, DateRange

import primegapverify



print("Setting up prime-records controller")

app = Flask(__name__)

SECRET_KEY = hex(random.randint(0, 2**32))
app.config["SECRET_KEY"] = SECRET_KEY

RECORDS_FN = "records.txt"
SUBMISSIONS_FN = "submissions.txt"
ALL_SQL_FN = "prime-gap-list/allgaps.sql"
GAPS_DB_FN = "prime-gap-list/gaps.db"

SQL_INSERT_PREFIX = "INSERT INTO gaps VALUES"
assert os.path.isfile(ALL_SQL_FN), "git init submodule first"

REALLY_LARGE = 10 ** 28000
SIEVE_PRIMES = 80 * 10 ** 6

# Globals for exchanging queue info with background thread
class WorkCoordinator():
    def __init__(self):
        # Status for server
        self.current = multiprocessing.Manager().list([None])

        # (threadsafe) Worker removes from this, Server pushes to this.
        self.queue = multiprocessing.Queue()

        # Used for client status
        self.client_queue = []

        # Used by server to tell client when items are done
        self.client_size = multiprocessing.Value('i', 0)

        # Worker pushes (without lock to this)
        self.recent = multiprocessing.Manager().list()

        # Client uses this to avoid double adding to queue (without checking)
        self.processed = {}

        # Records uploaded via this tool
        # list of traditional line format (gapsize,C?P,discovere,date,primedigit,start)
        new_records = []
        if os.path.isfile(RECORDS_FN):
            with open(RECORDS_FN, "r") as f:
                new_records = f.readlines()
        self.new_records = multiprocessing.Manager().list(new_records)

    def get_client_queue_lines(self):
        while len(self.client_queue) > self.client_size.value:
            self.client_queue.pop(0)

        return self.client_queue


# Worker thread
worker = None
global_coord = WorkCoordinator()


#--------------------


def gap_worker(coord):
    time.sleep(2)
    print("Gap Worker running")

    while True:
        coord.current[0] = None

        batch = coord.queue.get()
        verified = []
        for index, item in enumerate(batch, 1):
            gap_size, start, improved, line_fmt, sql_insert = item
            human = sql_insert[-1]
            discoverer = sql_insert[5]

            batch_message = "{} of {} ".format(index, len(batch))
            print("Worker running", batch_message, line_fmt)

            start_t = time.time()
            success, status = test_one(coord, gap_size, start, discoverer, human)
            end_t = time.time()

            coord.current[0] = "Postprocessing {}{}".format(
                batch_message if len(batch) > 1 else "", gap_size)

            if success:
                if success == 2:
                    print("Changing to C??")
                    assert "C?P" in line_fmt
                    line_fmt = line_fmt.replace("C?P", "C??")
                    assert sql_insert[2:5] == ("C", "?", "P"), sql_insert
                    sql_insert = sql_insert[:4] + ("?",) + sql_insert[5:]
                    item = (gap_size, start, improved, line_fmt, sql_insert)

                status = "Verified! ({:.1f}s)".format(end_t - start_t)
                verified.append(item)
                coord.recent.append((gap_size, human, status))
            else:
                coord.recent.append((gap_size, human, status))

            with coord.client_size.get_lock():
                coord.client_size.value -= 1

            print("       line:", line_fmt)

        if not verified:
            continue

        with open(ALL_SQL_FN, "r") as f:
            all_sql_lines = f.readlines()
            assert all_sql_lines, f"Empty {ALL_SQL_FN!r}"

        # process the batch into commits
        commit_msgs = []
        new_records = 0
        improved_merit = 0
        # Handles multiple authors
        authors = set()
        for gap_size, _, improved, line_fmt, sql_insert in verified:

            # Write to allgaps.sql (sorted), kinda slow
            replace = update_all_sql(all_sql_lines, gap_size, sql_insert)
            new_records += not replace
            improved_merit += improved

            commit_msg = "{} record {} merit={} found by {}".format(
                "Improved" if replace else "New",
                sql_insert[0], sql_insert[7], sql_insert[5])
            commit_msgs.append(commit_msg)
            authors.add(sql_insert[5])

        # Write to record file
        with open(RECORDS_FN, "a") as f:
            for _, _, _, line_fmt, _ in verified:
                f.write(line_fmt + "\n")

        # Write lines back to file
        with open(ALL_SQL_FN, "w") as f:
            for line in all_sql_lines:
                f.write(line)

        if len(commit_msgs) > 1:
            updates = len(verified) - new_records
            header = "{}{}by {} (merit +{:.2f}) gaps {} to {}\n".format(
                "{} updates ".format(updates) if updates else "",
                "{} new gaps ".format(new_records) if new_records else "",
                min(authors) if len(authors) == 1 else "various",
                improved_merit,
                min(item[0] for item in verified),
                max(item[0] for item in verified))
            commit_msgs.insert(0, header)

        commit_msg = "\n".join(commit_msgs)
        git_commit_and_push(commit_msg)

        # Save to new_records now that batch is finished
        for _, _, _, line_fmt, _ in verified:
            coord.new_records.append(line_fmt)

        # Write to gaps.db
        with open_db() as db:
            start_count = list(db.execute("SELECT COUNT(*) FROM gaps"))
            for gap_size, _, _, line_fmt, sql_insert in verified:
                # Delete any existing gap
                a = db.execute("DELETE FROM gaps WHERE gapsize = ?", (gap_size,))
                # Insert new gap into db
                b = db.execute(SQL_INSERT_PREFIX + str(sql_insert))
                if a.rowcount != 1 or b.rowcount != 1:
                    print ("UPDATE {} UNEQUAL: {} & {}".format(
                        sql_insert, a.rowcount, b.rowcount))
            db.commit()
            end_count = list(db.execute("SELECT COUNT(*) FROM gaps"))

        print ("Batch Done! @{} (count {} => {})".format(
            datetime.datetime.now().isoformat(sep=' '),
            start_count[0], end_count[0]))


def sieve_interval(human, start, gap_size, faster):
    # if human is "standard" form use `large_sieve` from
    # https://github.com/sethtroisi/prime-gap-verify
    # TODO: try to share some python parsing code too.

    parsed = primegapverify.parse_primorial_standard_form(human)
    if parsed:
        m, p, d, a = parsed
        assert p in range(30, 80000), p
        assert m in range(2 ** 62), m
        assert d in range(2 ** 62), d
        assert -10 ** 7 <= a < 0, a

    num = primegapverify.parse(human)
    assert num, "should be parsable: " + human

    N_bits = float(gmpy2.log2(num))
    max_prime = primegapverify.sieve_limit(N_bits, gap_size)
    if faster:
        max_prime //= 5

    t0 = time.time()
    composite = primegapverify.sieve(num, gap_size, max_prime=max_prime)
    t1 = time.time()
    assert len(composite) == gap_size + 1
    return composite, t1 - t0


def test_one(coord, gap_size, start, discoverer, human):
    tests = 0
    coord.current[0] = "Testing {}".format(gap_size)

    prime_test_time = time.time()
    if not primegapverify.is_prime_large(start):
        return False, "start not prime"

    if not primegapverify.is_prime_large(start + gap_size):
        return False, "end not prime"
    prime_test_time = time.time() - prime_test_time

    verified_type = 1

    assert start % 2 == 1

    tests += 2
    coord.current[0] = "Testing {}, Endpoints tested, sieving interval".format(
        gap_size)

    log_n = gmpy2.log(start)
    merit = gap_size / log_n
    test_fraction = 1

    # Primes take ~4.5x longer (for gmpy2, but not pfgw64)
    if gmpy2.num_digits(start, 2) <= 8000:
        prp_time = prime_test_time / 2 / 4.8
    else:
        prp_time = prime_test_time / 2

    faster = (merit < 22) and (log_n > 6000)
    composite, sieve_time = sieve_interval(human, start, gap_size, faster)
    # Endpoints have been verified prime, something is very wrong.
    assert composite[0] is False and composite[-1] is False

    prp_count = composite.count(False)
    expected_time = prp_time * prp_count

    print ("       {:30s} | {:5} PRP, expected time {:4.1f} (ends: {:.1f}, sieve: {:.1f})".format(
        human, prp_count, expected_time, prime_test_time, sieve_time))

    if expected_time > 3 * 60 and merit < 22:
        print ("MEGAGAP: log_n: {:.0f} (estimated prp_time: {:.1f} = {:.2f}s x ~{}) "
               "merit: {:.1f}".format(
            log_n, expected_time, prp_time, prp_count, merit))

        if discoverer in ("Jacobsen", "Rosnthal", "M.Jansen",
                          "MrtnRaab", "RobSmith", "S.Troisi"):
            # This discoverer's are trusted
            verified_type = 1
            test_fraction = 1 * 60 / expected_time
        else:
            # Change to C??
            verified_type = 2
            test_fraction = 10 * 60 / expected_time

    if test_fraction > 0.8:
        test_fraction = 1.0

    composites = 2 # for endpoints
    for k in range(2, gap_size, 2):
        if composite[k]: continue
        composites += 1

        if test_fraction > 0 and random.random() > test_fraction:
            continue

        if primegapverify.is_prime_large(start + k):
            return False, "start + {} is prime".format(k)

        tests += 1
        coord.current[0] = "Testing {}, {}/{} done, {}/{} PRPs performed".format(
            gap_size, k+1, gap_size, tests, composites)

    return verified_type, "Verified"


def update_all_sql(all_sql_lines, gap_size, sql_insert):
    assert 110 < len(all_sql_lines) < 110000, len(all_sql_lines)

    # HACK: want merit to be like 15.230 not 15.23 so used {:.3f} not round
    # Downside is str(sql_insert) changes that to '15.230' so convert back here.
    str_sql_insert = str(sql_insert).replace(" ", "")
    str_sql_insert = re.sub(r",'([0-9.]+)',", r",\1,", str_sql_insert)
    new_line = SQL_INSERT_PREFIX + str_sql_insert + ";\n"

    # TODO: Could start near line 2*gaps+1
    LINE_PREFIX = "INSERT INTO gaps"

    # Find the right place in the insert section
    for index, line in enumerate(all_sql_lines):
        if line.startswith(LINE_PREFIX):
            # Get the first number from the line
            match = re.search(r"\(([0-9]+),", line)
            assert match
            gap = int(match.group(1))
            if gap >= gap_size:
                break

    # Format is wrong
    assert 100 < index, ("SMALL INDEX", index, new_line)
    assert index+10 < len(all_sql_lines), ("LARGE INDEX", index, new_line)
    assert LINE_PREFIX in all_sql_lines[index+1], ("LARGE INDEX", index, new_line)

    start_insert_line = SQL_INSERT_PREFIX + "(" + str(gap_size)
    replace = start_insert_line in all_sql_lines[index]
    if replace:
        print ("  Replacing ", all_sql_lines[index].strip())
        print ("  With      ", new_line.strip())
        all_sql_lines[index] = new_line
    else:
        print (" !INSERTING!", new_line.strip())
        all_sql_lines.insert(index, new_line)

    # Possible doesn't work if two updates to same gap
    return replace


def git_commit_and_push(commit_msg):
    # TODO verify git is clean before
    # TODO verify commit has correct number of inserts / deletes

    wd = os.getcwd()
    try:
        os.chdir("prime-gap-list")

        # TODO verify we're on branch "server"

        subprocess.check_output(["git", "commit", "-am", commit_msg])
        subprocess.check_output(["git", "push", "upstream", "server:server"])
        print()
    except Exception as e:
        print("Error!", e)
    os.chdir(wd)


def open_db():
    return sqlite3.connect(GAPS_DB_FN)


def get_db():
    db = getattr(g, "_database", None)
    # Setup
    if db is None:
        db = g._database = open_db()
    db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def short_start(n):
    str_n = str(n)
    if len(str_n) < 20:
        return str_n
    return "{}...{}<{}>".format(str_n[:3], str_n[-3:], len(str_n))


def possible_add_to_queue(
        coord,
        gap_size, gap_start,
        discoverer,
        discover_date):

    # Check gapsize odd, >1200
    if gap_size % 2 == 1:
        return False, "Odd gapsize is unlikely"
    if gap_size <= 1200:
        return False, "optimal gapsize={} has already been found".format(
            gap_size)

    # Parse gap_start, check odd, size
    start_n = primegapverify.parse(gap_start)
    if start_n is None:
        err_msg = ("Can't parse gapstart={} (post on "
                   "MersenneForum if this is an error)").format(gap_start)
        return False, err_msg
    if start_n >= REALLY_LARGE:
        return False, "gapstart={} is to large at this time".format(gap_start)
    if start_n % 2 == 0:
        return False, "gapstart={} is even".format(gap_start)

    # Check if already queued (or better queued)
    existing = coord.processed.get(gap_size)
    if existing:
        if existing < start_n:
            return False, "Existing better record submitted"
        elif existing == start_n:
            return False, "Already processed"

    coord.processed[gap_size] = start_n

    # Check if DB for current record
    with get_db() as db:
        rv = list(db.execute("SELECT merit, startprime FROM gaps WHERE gapsize = ?", (gap_size,)))
        assert len(rv) in (0, 1), [tuple(r) for r in rv]
        # existing_merit_in_db, existing_start_prime
        if len(rv) == 1:
            e_merit_db, e_startprime = rv[0]
            e_start = primegapverify.parse(e_startprime)
            if e_start:
                e_merit = gap_size / gmpy2.log(e_start)
                if abs(e_merit_db - e_merit) > 0.01:
                    assert False, ("Bad record merit for gap:", gap_size, e_merit_db, e_merit)
            else:
                e_merit = e_merit_db
                print(f"Unable to process startprime={e_startprime}")
        else:
            e_start = None
            e_merit = 0

    new_merit = gap_size / gmpy2.log(start_n)
    improved_merit = new_merit - e_merit

    if e_start is not None:
        if start_n == e_start:
            return False, "Already submitted"
        elif start_n > e_start:
            return False, "Existing gap {} with better {} {:.4f} vs {} {:.4f}".format(
                gap_size,
                short_start(e_start),
                e_merit,
                short_start(start_n),
                new_merit)

    # Old style line & github entry
    newmerit_fmt = "{:.4f}".format(new_merit)
    primedigits = len(str(start_n))

    # Can be changed to C?? in test_one
    # See https://primegap-list-project.github.io/drtrnicely-format-legacy/
    # C = Common, Classic
    # ? = Not first occurrence
    # P = probabilistic primes (at endpoints)
    ccc_type = "C?P"

    line_fmt = "{}, {}, {}, {}, {}, {}, {}".format(
        gap_size, ccc_type, newmerit_fmt, discoverer,
        discover_date.isoformat(), primedigits, gap_start)

    year = discover_date.year
    assert 2005 <= year <= 2030, year
    sql_insert = (gap_size, 0) + tuple(ccc_type) + (discoverer,
        year, newmerit_fmt, primedigits, gap_start)

    item = (gap_size, start_n, improved_merit, line_fmt, sql_insert)

    return item, "Adding {} gapsize={} to queue, would improve merit {} to {:.3f}".format(
        short_start(start_n), gap_size,
        "{:.3f}".format(e_merit) if e_merit > 0.1 else "NONE",
        new_merit)


def possible_add_to_queue_log(coord, form):
    discoverer = form.discoverer.data
    is_log_discoverer = discoverer.lower() in ("file", "log")
    log_data = form.logdata.data
    # Not yet checked for > previous records
    line_datas = []
    statuses = []

    # Numbers of the form {m*P#/d-s, m*P#/d#-s, P#/d-s, b^p-a}
    number_re = r"((\d+\s*\*\s*)?\d+#\s*/\s*\d+#?\s*\-\s*\d+|\d+\^\d+-\d+|\d+)"

    for line in log_data.split("\n"):
        if len(line.strip()) == 0:
            continue

        # <gap> <YYYY-MM-DD> <Namedate e.g. 2020-05-23> <name> <merit> <START>
        full_log_re = (r"(\d+)\s+"
                       r"(20[012]\d-\d\d?-\d\d?)\s+"
                       r"([\w.]+)\s+"
                       r"([\d.]+)\s+") + number_re
        match = re.search(full_log_re, line)
        if match:
            if is_log_discoverer:
                line_date = datetime.datetime.strptime(match.group(2), "%Y-%m-%d").date()
                line_datas.append((int(match.group(1)), match.group(5), match.group(3), line_date))
                continue
            else:
                statuses.append("Must set discoverer=log to use full line format")
                continue

        if is_log_discoverer:
            statuses.append("Must set real discoverer")
            continue

        # <gap> <experected_merit> <START>
        log_re = r"(\d+)\s+([\d.]+)\s+" + number_re
        match = re.search(log_re, line)
        if match:
            discover_date = form.date.data
            line_datas.append((int(match.group(1)), match.group(3), discoverer, discover_date))
            continue

        gap_start_re = r"(\d+)\s+" + number_re
        match = re.search(gap_start_re, line)
        if match:
            discover_date = form.date.data
            line_datas.append((int(match.group(1)), match.group(2), discoverer, discover_date))
            continue

        statuses.append("Didn't find match in: " + line)

    with open(SUBMISSIONS_FN, "a") as f:
        f.write("BATCH\n")
        for size, start, who, when in line_datas:
            f.write(f"{size} {when} {who} 1.234 {start}\n")
        f.write("\n\n")

    batch = []
    for size, start, who, when in line_datas:
        item, status = possible_add_to_queue(coord, size, start, who, when)
        if item:
            batch.append(item)
        statuses.append(status)

    if batch:
        with coord.client_size.get_lock():
            coord.client_size.value += len(batch)
            coord.queue.put(batch)
            for item in batch:
                coord.client_queue.append(item[3])

    print("Processed {} lines to {} batch".format(
        len(line_datas), len(batch)))

    return len(batch) > 0, "\n<br>\n".join(statuses)


class GapLogForm(FlaskForm):
    @staticmethod
    def valid_date_range():
        return DateRange(
        min=datetime.date(2015, 1, 1),
        max=datetime.date.today() + datetime.timedelta(hours=48))

    discoverer = StringField(
        "Discoverer (or log if present in log)",
        render_kw={"size": 12},
        validators=[DataRequired(), Length(min=3, max=8)])

    date = DateField(
        "Date",
        validators=[DataRequired(), DataRequired()])

    logdata = TextAreaField(
        "LogData",
        description=(
            "206048  20.785  100017163 * 10007#/30030 -138324 to +67724\n"
            "or\n"
            "22558   23.779  104304433*977#/7#-15234\n"
            "or\n"
            "2074 2019-09-25 M.Jansen 28.311600 3513398427*71#/30030 - 1532\n"
        ),
        render_kw={"rows": 10},
        validators=[DataRequired()])

    submit = SubmitField("Add")

    @classmethod
    def new(csl):
        form = csl()
        form.date.validators[0] = GapLogForm.valid_date_range()
        return form



@app.route("/", methods=("GET", "POST"))
def controller():
    global global_coord
    coord = global_coord

    formB = GapLogForm.new()
    which_form = request.args.get("form")

    status = ""
    added = False
    if which_form is not None and coord.queue.qsize() > 1000:
        return "Long queue try again later"

    if formB.validate_on_submit():
        added, status = possible_add_to_queue_log(coord, formB)

    queue_data = coord.get_client_queue_lines()
    queued = len(queue_data)

    return render_template(
        "record-check.html",
        formB=formB,
        status=status,
        queued=queued,
        queue=queue_data,
    )


@app.route("/status")
def status():
    global global_coord, worker
    coord = global_coord

    queue_data = coord.get_client_queue_lines()
    queued = len(queue_data)
    return render_template(
        "status.html",
        running=worker.is_alive(),
        new_records_count=len(coord.new_records),
        new_records=coord.new_records[-100:],
        recent=coord.recent[-50:],
        queue=queue_data,
        queued=queued)


@app.route("/validate-gap")
def stream():
    def gap_status_stream():
        global global_coord
        coord = global_coord
        if coord.client_size.value == 0:
            return

        for i in range(3600):
            queued = coord.client_size.value
            if not queued:
                break

            state = "Queue {}: {}".format(queued, coord.current[0])
            yield "data: " + state + "\n\n"
            time.sleep(5)

        yield "data: Done (refresh to see recent status)\n\n"

    return Response(gap_status_stream(), mimetype="text/event-stream")


@app.route("/merits.txt")
def merits():
    # Write merits.txt
    merits = list(get_db().execute(
        "SELECT gapsize, merit, discoverer FROM gaps ORDER BY gapsize ASC"))

    rows = []
    for m in merits:
        rows.append("{} {:.4f} {}".format(*m))

    return Response("\n".join(rows), mimetype="text/plain")


@app.route("/graph_merit.csv")
def graph_merit_csv():
    max_gap = request.args.get("max")
    if max_gap and max_gap.isdigit():
        max_gap = int(max_gap)
    else:
        max_gap = 10000

    merits = get_db().execute(
        "SELECT gapsize, merit, discoverer FROM gaps WHERE gapsize < ? ORDER BY gapsize ASC",
        (min(120000, max_gap),))
    return Response(
        "gapsize,merit,discoverer\n" +
            "\n".join(",".join(map(str, row)) for row in merits),
        mimetype="text/plain"
    )


@app.route("/graphs")
@app.route("/graphs.html")
def graphs():
    return app.send_static_file("graphs.html")


@app.route("/secret_test")
def secret_test_page():
    status = "<br>".join([
        secret_test_page_write_db(),
        secret_test_page_test_one(),
    ])
    return status

@app.route("/secret_test_write_db")
def secret_test_page_write_db():
    # Write to secret table in DB
    try:
        with get_db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS testing ("
                "  five INTEGER PRIMARY KEY,"
                "  time INTEGER)")
            db.execute("INSERT OR IGNORE INTO testing VALUES(5, 1)")
            last = db.execute("SELECT * FROM testing")
            updated = int(time.time())
            db.execute(f"UPDATE testing SET time={updated}")
            db.commit()
    except sqlite3.OperationalError as e:
        import traceback
        print ("Error on secret test page")
        traceback.print_exc()
        return str(e)

    return "Updated=" + str(updated)

@app.route("/secret_test_test_records")
def secret_test_page_test_one():
    global global_coord
    coord = global_coord

    # Test with a couple of small records
    output = []
    try:
        with get_db() as db:
            lines = list(tuple(r) for r in db.execute(
                "SELECT gapsize, startprime, discoverer FROM gaps "
                "WHERE gapsize BETWEEN 100 and 5000 AND primedigits < 100"))
            lines = sorted(random.sample(lines, 100))
            output.append("Sampled 100 lines first: " + str(lines[0]))

            for gap_size, human, who in lines:
                start = primegapverify.parse(human)
                if not start:
                    output.append("Failed to parse: " + human)
                else:
                    success, status = test_one(coord, gap_size, start, who, human)
                    if not success:
                        output.append("Failed({}): {}".format(status, (gap_size, human)))
    except Exception as e:
        import traceback
        print ("Error on secret test page")
        traceback.print_exc()
        return "Exception: " + str(e)

    return "<br>".join(output)


# Create background gap_worker
worker = multiprocessing.Process(target=gap_worker, args=(global_coord,))
worker.start()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        # host = "::",
        port=5090,
        debug=False,
        # debug=True,
    )

    worker.terminate()
