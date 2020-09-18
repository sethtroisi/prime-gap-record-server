import datetime
import os
import random
import re
import multiprocessing
import subprocess
import sqlite3
import time

import gmpy2
from flask import Flask, Response, g, render_template, request
from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired
from wtforms_components import IntegerField, DateField, DateRange


print("Setting up prime-records controller")

app = Flask(__name__)

SECRET_KEY = hex(random.randint(0, 2**32))
app.config["SECRET_KEY"] = SECRET_KEY

RECORDS_FN = "records.txt"
ALL_SQL_FN = "prime-gap-list/allgaps.sql"
GAPS_DB_FN = "gaps.db"

SQL_INSERT_PREFIX = "INSERT INTO gaps VALUES"
assert os.path.isfile(ALL_SQL_FN), "git init submodule first"

REALLY_LARGE = 10 ** 28000
SIEVE_PRIMES = 40 * 10 ** 6


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
        self.processed = set()

        # Records uploaded via this tool
        # list of traditional line format (gapsize,C?P,discovere,date,primedigit,start)
        self.new_records = []
        if os.path.isfile(RECORDS_FN):
            with open(RECORDS_FN, "r") as f:
                self.new_records = f.readlines()

    def get_client_queue_lines(self):
        while len(self.client_queue) > self.client_size.value:
            self.client_queue.pop(0)

        return self.client_queue


# Worker thread
worker = None
global_coord = WorkCoordinator()


#--------------------


def gap_worker(coord):
    time.sleep(4)
    print("Gap Worker running")

    while True:
        coord.current[0] = None

        batch = coord.queue.get()
        verified = []
        for index, item in enumerate(batch, 1):
            gap_size, start, _, line_fmt, sql_insert = item
            human = sql_insert[-1]

            batch_message = "{} of {} ".format(index, len(batch))
            print("Worker running", batch_message, line_fmt)

            start_t = time.time()
            discoverer = sql_insert[5]
            success, status = test_one(coord, gap_size, start, discoverer)
            end_t = time.time()

            coord.current[0] = "Postprocessing {}{}".format(
                batch_message if len(batch) > 1 else "", gap_size)

            if success:
                status = "Verified! ({:.1f}s)".format(end_t - start_t)
                verified.append(item)
                coord.recent.append((gap_size, human, status))
                coord.new_records.append(line_fmt)
            else:
                coord.recent.append((gap_size, human, status))

            with coord.client_size.get_lock():
                coord.client_size.value -= 1

            print("       line:", line_fmt)

        if not verified:
            continue

        # process the batch into commit, records.
        commit_msgs = []
        new_records = 0
        improved_merit = 0
        for gap_size, _, improved, line_fmt, sql_insert in verified:

            # Write to record file
            with open(RECORDS_FN, "a") as f:
                f.write(line_fmt + "\n")

            # Write to allgaps.sql (sorted), kinda slow
            replace = update_all_sql(gap_size, sql_insert)
            new_records += not replace
            improved_merit += improved

            commit_msg = "{} record {} merit={} found by {}".format(
                "Improved" if replace else "New",
                sql_insert[0], sql_insert[7], discoverer)
            commit_msgs.append(commit_msg)

        if len(commit_msgs) > 1:
            # TODO What to do if multiple authors?
            header = "{} updates {}by {} (merit +{:.2f}) gaps {} to {}\n".format(
                len(commit_msgs) - new_records,
                "{} new gaps ".format(new_records) if new_records else "",
                discoverer,
                improved_merit,
                min(item[0] for item in verified),
                max(item[0] for item in verified))
            commit_msgs.insert(0, header)

        commit_msg = "\n".join(commit_msgs)
        git_commit_and_push(commit_msg)

        # Write to gaps.db
        with open_db() as db:
            start_count = list(db.execute("SELECT COUNT(*) FROM gaps"))
            for gap_size, _, _, _, sql_insert in verified:
                # Delete any existing gap
                a = db.execute("DELETE FROM gaps WHERE gapsize = ?", (gap_size,))
                # Insert new gap into db
                b = db.execute(SQL_INSERT_PREFIX + str(sql_insert))
                if a.rowcount != 1 or b.rowcount != 1:
                    print ("UPDATE {} FAILED: {} & {}".format(
                        sql_insert, a.rowcount, b.rowcount))
            db.commit()
            end_count = list(db.execute("SELECT COUNT(*) FROM gaps"))

        print ("Batch Done! @{} (count {} => {})".format(
            datetime.datetime.now().isoformat(sep=' '),
            start_count[0], end_count[0]))


def test_one(coord, gap_size, start, discoverer):
    tests = 0
    coord.current[0] = "Testing {}".format(gap_size)

    if not gmpy2.is_prime(start):
        return False, "start not prime"

    if not gmpy2.is_prime(start + gap_size):
        return False, "end not prime"

    assert start % 2 == 1

    tests += 2
    coord.current[0] = "Testing {}, {}/{} done, {} PRPs performed".format(
        gap_size, 2, gap_size, tests)

    # TODO use gmpy2.next_prime()
    #   pros: faster (~2x)
    #   cons: no status (maybe create timestamp somewhere)

    log_n = gmpy2.log(start)
    sieve_primes = min(1000, min(SIEVE_PRIMES, log_n ** 2))

    composite = [False for i in range(gap_size+1)]
    primes = [True for i in range(sieve_primes//2+1)]
    for p in range(3, sieve_primes):
        if not primes[p//2]: continue

        # Sieve other primes
        for m in range(p*p//2, sieve_primes//2+1, p):
            primes[m] = False

        # Remove any numbers in the interval divisible by p
        first = (-start) % p
        for m in range(first, gap_size+1, p):
            # assert (start + m) % p == 0
            composite[m] = True

    # Endpoints have been verified prime, something is very wrong.
    assert composite[0] is False and composite[-1] is False

    # TODO: Do something better here based on name, size...
    merit = gap_size / log_n
    skip_fraction = 0

    if log_n > 20000 and merit < 20:
        # Megagap verification
        if discoverer in ("MrtnRaab",):
            # Tries to keep total verification time under a couple of hours
            prp_time = log_n ** 2.5 / 2e10
            prp_count = gap_size / (log(sieve_primes) * exp(0.57721))

            skip_fraction = (3600 / prp_time) / prp_count
            print ("MEGAGAP: log_n: {} (estimated prp_time: {:.0f}s x {:.0f}) "
                   "merit: {:.1f} skip_fraction: {:.4f} (1 in {:.1f})".format(
                log_n, prp_time, prp_count, prp_count,
                merit, skip_fraction, 1 / skip_fraction))

    elif log_n > 5000 and merit < 20:
        # More trusted discoverers
        if discoverer in ("Jacobsen", "M.Jansen", "RobSmith", "Rosnthal", "MrtnRaab"):
            skip_fraction = 0.7

    for k in range(2, gap_size, 2):
        if composite[k]: continue

        if skip_fraction > 0 and random.random() < skip_fraction:
            continue

        if gmpy2.is_prime(start + k):
            return False, "start + {} is prime".format(k)

        tests += 1
        coord.current[0] = "Testing {}, {}/{} done, {} PRPs performed".format(
            gap_size, k+1, gap_size, tests)

    return True, "Verified"


def update_all_sql(gap_size, sql_insert):
    sql_lines = []
    with open(ALL_SQL_FN, "r") as f:
        sql_lines = f.readlines()

    assert 110 < len(sql_lines) < 100000, len(sql_lines)

    # HACK: want merit to be like 15.230 not 15.23 so used {:.3f} not round
    # Downside is str(sql_insert) changes that to '15.230' so convert back here.
    str_sql_insert = str(sql_insert).replace(" ", "")
    str_sql_insert = re.sub(r",'([0-9.]+)',", r",\1,", str_sql_insert)
    new_line = SQL_INSERT_PREFIX + str_sql_insert + ";\n"

    # Find the right place in the insert section
    for index, line in enumerate(sql_lines):
        if line.startswith("INSERT INTO gaps"):
            # Get the first number from the line
            match = re.search(r"\(([0-9]+),", line)
            assert match
            gap = int(match.group(1))
            if gap >= gap_size:
                break

    # Format is wrong
    assert (100 < index < 94000), ("WEIRD INDEX", index, new_line)

    start_insert_line = SQL_INSERT_PREFIX + "(" + str(gap_size)
    replace = start_insert_line in sql_lines[index]
    if replace:
        print ("  Replacing ", sql_lines[index].strip())
        print ("  With      ", new_line.strip())
        sql_lines[index] = new_line
    else:
        sql_lines.insert(index, new_line)

    # Write file back out (slow to do each time, but unavoidable?)
    with open(ALL_SQL_FN, "w") as f:
        for line in sql_lines:
            f.write(line)

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


NUMBER_RE_1 = re.compile(r"^(\d+)\*\(?(\d+)#\)?/(\d+)-(\d+)$")
NUMBER_RE_2 = re.compile(r"^\(?(\d+)#\)?/(\d+)-(\d+)$")
NUMBER_RE_3 = re.compile(r"^(\d+)\*(\d+)#/\((\d+)#\*(\d+)\)-(\d+)$")
NUMBER_RE_4 = re.compile(r"^(\d+)\*(\d+)#/\((\d+)\*(\d+)\)-(\d+)$")


def parse_num_fast(start):
    start = start.replace(" ", "")
    num = None
    if start.isdigit():
        return int(start)

    num_match = NUMBER_RE_1.match(start)
    if num_match:
        m, p, d, a = map(int, num_match.groups())
        K = m * gmpy2.primorial(p)
        assert K % d == 0
        return K // d - a

    num_match = NUMBER_RE_2.match(start)
    if num_match:
        p, d, a = map(int, num_match.groups())
        K = gmpy2.primorial(p)
        assert K % d == 0
        return K // d - a

    num_match = NUMBER_RE_3.match(start)
    if num_match:
        m, p, d1, d2, a = map(int, num_match.groups())
        K = m * gmpy2.primorial(p)
        D = gmpy2.primorial(d1) * d2
        assert K % D == 0
        return K // D - a

    num_match = NUMBER_RE_4.match(start)
    if num_match:
        m, p, d1, d2, a = map(int, num_match.groups())
        K = m * gmpy2.primorial(p)
        D = d1 * d2
        assert K % D == 0
        return K // D - a

    return None

def parse_num(num_str):
    fast = parse_num_fast(num_str)
    if fast:
        return fast

    if "(" in num_str or ")" in num_str:
        return None
    if len(num_str) > 10000:
        return None

    # Remove whitespace from string
    num_str = re.sub(r"\s", "", num_str)

    # Generally follows this form
    match = re.search(r"(\d+)\*(\d+)#/(\d+)(#?)\-(\d+)", num_str)
    if match:
        if any(len(v) > 100 for v in match.groups()):
            return REALLY_LARGE
        m, p, d, dp, a = match.groups()
        m, p, d, a = map(int, (m, p, d, a))

        # Check if p will be > REALLY_LARGE
        if p > 100000:
            return REALLY_LARGE
        if p < 50:
            return None

        d = d
        if dp == '#':
            d = gmpy2.primorial(d)

        t = m * gmpy2.primorial(p)
        if t % d != 0:
            return None

        return int(t // d - a)
    return None


def possible_add_to_queue(
        coord,
        gap_size, gap_start,
        ccc_type, discoverer,
        discover_date):
    if gap_size % 2 == 1:
        return False, "Odd gapsize is unlikely"
    if gap_size <= 1200:
        return False, "optimal gapsize={} has already been found".format(
            gap_size)

    gap_start = gap_start.replace(" ", "")
    if any(gap_start in line for line in coord.new_records):
        return False, "Already added to records"

    if (gap_size, gap_start) in coord.processed:
        return False, "Already processed"
    coord.processed.add((gap_size, gap_start))

    rv = list(get_db().execute(
        "SELECT merit, startprime FROM gaps WHERE gapsize = ?",
        (gap_size,)))
    assert len(rv) in (0, 1), [tuple(r) for r in rv]
    # existing_merit_in_db, existing_start_prime
    if len(rv) == 1:
        e_merit_db, e_startprime = rv[0]
        e_start = parse_num(e_startprime)
    else:
        e_start = REALLY_LARGE
        e_merit_db = gap_size / gmpy2.log(e_start)

    e_merit = gap_size / gmpy2.log(e_start)
    if abs(e_merit_db - e_merit) > 0.01:
        assert False, ("Bad record merit for gap:", gap_size, e_merit_db, e_merit)

    start_n = parse_num(gap_start)
    if start_n is None:
        err_msg = ("Can't parse gapstart={} (post on "
                   "MersenneForum if this is an error)").format(gap_start)
        return False, err_msg
    if start_n >= REALLY_LARGE:
        return False, "gapstart={} is to large at this time".format(gap_start)
    if start_n % 2 == 0:
        return False, "gapstart={} is even".format(gap_start)

    new_merit = gap_size / gmpy2.log(start_n)
    if start_n >= e_start:
        return False, "Existing gap {} with better {} {:.3f} vs {} {:.4f}".format(
            gap_size,
            short_start(e_start),
            e_merit,
            short_start(start_n),
            new_merit)

    improved_merit = new_merit - e_merit

    # Pull data from form for old style line & github entry
    newmerit_fmt = "{:.3f}".format(new_merit)
    primedigits = len(str(start_n))

    line_fmt = "{}, {}, {}, {}, {}, {}, {}".format(
        gap_size, ccc_type, newmerit_fmt, discoverer,
        discover_date.isoformat(), primedigits, gap_start)

    year = discover_date.year
    assert 2015 <= year <= 2024, year
    sql_insert = (gap_size, 0) + tuple(ccc_type) + (discoverer,
        year, newmerit_fmt, primedigits, gap_start)

    item = (gap_size, start_n, improved_merit, line_fmt, sql_insert)

    return item, "Adding {} gapsize={} to queue, would improve merit {} to {:.3f}".format(
        short_start(start_n), gap_size,
        "{:.3f}".format(e_merit) if e_merit > 0.1 else "NONE",
        new_merit)


def possible_add_to_queue_log(coord, form):
    discoverer = form.discoverer.data
    log_data = form.logdata.data

    # How to choice this better?
    ccc_type = "C?P" # TODO describe this somewhere

    # Not yet checked for > previous records
    line_datas = []
    statuses = []

    #Primorial in the form {m*P#/d-s, m*P#/d#-s, P#/d-s}
    primorial_re = r"((\d+\s*\*\s*)?\d+#\s*/\s*\d+#?\s*\-\s*\d+)"

    for line in log_data.split("\n"):
        if len(line.strip()) == 0:
            continue

        # TODO example
        full_log_re = (r"(\d+)\s+"
                       r"(20[12]\d-\d\d?-\d\d?)\s+([\w.]+\.?[\w.]*)\s+"
                       r"([\d.]+)\s+") + primorial_re
        match = re.search(full_log_re, line)
        if match:
            if discoverer.lower() not in ("file", "log"):
                statuses.append("Must set discoverer=log to use full line format")
                continue
            else:
                line_date = datetime.datetime.strptime(match.group(2), "%Y-%m-%d").date()
                line_datas.append((int(match.group(1)), match.group(5), match.group(3), line_date))
                continue

        # <gap> <experected_merit> <START>
        log_re = r"(\d+)\s+([\d.]+)\s+" + primorial_re
        match = re.search(log_re, line)
        if match:
            discover_date = form.date.data
            line_datas.append((int(match.group(1)), match.group(3), discoverer, discover_date))
            continue

        gap_start_re = r"(\d+)\s+" + primorial_re
        match = re.search(gap_start_re, line)
        if match:
            discover_date = form.date.data
            line_datas.append((int(match.group(1)), match.group(2), discoverer, discover_date))
            continue

        statuses.append("Didn't find match in: " + line)

    batch = []
    for size, start, who, when in line_datas:
        item, status = possible_add_to_queue(coord, size, start, ccc_type, who, when)
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
        validators=[DataRequired()])

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
        new_records=coord.new_records,
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


if __name__ == "__main__":
    # Create background gap_worker
    worker = multiprocessing.Process(target=gap_worker, args=(global_coord,))
    worker.start()

    app.run(
        host="0.0.0.0",
        # host = "::",
        port=5090,
        debug=False,
        # debug=True,
    )

    worker.terminate()

