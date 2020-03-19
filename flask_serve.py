import datetime
import os
import random
import re
import subprocess
import sqlite3
import threading
import time
from queue import Queue

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

REALLY_LARGE = 10 ** 10000
SIEVE_PRIMES = 40 * 10 ** 6


# Globals for exchanging queue info with background thread

# Worker thread
worker = None

# gap_size, start, batch_num, line_fmt, sql_insert
queue = Queue()

# current status
current = None

# list of (gapsize, human-readable start, status)
recent = []

# Records uploaded via this tool
# list of traditional line format (gapsize,C?P,discovere,date,primedigit,start)
new_records = []
if os.path.isfile(RECORDS_FN):
    with open(RECORDS_FN, "r") as f:
        new_records = f.readlines()

#--------------------


def gap_worker():
    global new_records, recent, queue, current

    print("Gap Worker running")

    batch_messages = []

    while True:
        current = None
        # Blocks without busy wait
        queue.not_empty.acquire()
        if queue._qsize() == 0:
            print("Worker waiting for not_empty")
            queue.not_empty.wait()
            # Helps with grouping batch
            time.sleep(0.1)

        queue_0 = queue.queue[0]
        queue.not_empty.release()

        batch_num, gap_size, start, line_fmt, sql_insert = queue_0
        human = sql_insert[-1]

        start_t = time.time()
        discoverer = sql_insert[5]
        success, status = test_one(gap_size, start, discoverer)
        end_t = time.time()

        popped = queue.get()
        assert popped == queue_0

        more_in_batch = len(queue.queue) > 0 and queue.queue[0][0] == batch_num

        current = "Postprocessing{} {}".format(
            " batch" if more_in_batch else "",
            gap_size)

        if not success:
            recent.append((gap_size, human, status))
        else:
            status = "Verified! ({:.1f}s)".format(end_t - start_t)
            recent.append((gap_size, human, status))
            new_records.append(line_fmt)

            print("       line:", line_fmt)

            # Write to record file
            with open(RECORDS_FN, "a") as f:
                f.write(line_fmt + "\n")

            # Write to allgaps.sql (sorted), kinda slow
            replace = update_all_sql(sql_insert)

            commit_msg = "{} record {} merit={} found by {}".format(
                "Improved" if replace else "New",
                sql_insert[0], sql_insert[7], discoverer)
            batch_messages.append(commit_msg)

            if not more_in_batch:
                if len(batch_messages) == 1:
                    commit_msg = batch_messages[0]
                else:
                    # TODO What to do if author not the same
                    header = "{} Records | First {}\n".format(
                        len(batch_messages), batch_messages[0])
                    commit_msg = "\n".join([header] + batch_messages)
                git_commit_and_push(commit_msg)
                batch_messages = []
                print("\n")

            # Write to gaps.db
            with open_db() as db:
                # Delete any existing gap
                db.execute("DELETE FROM gaps WHERE gapsize = ?", (gap_size,))
                # Insert new gap into db
                db.execute(SQL_INSERT_PREFIX + str(sql_insert))
                db.commit()


def test_one(gap_size, start, discoverer):
    global current

    tests = 0
    current = "Testing {}".format(gap_size)

    if not gmpy2.is_prime(start):
        return False, "start not prime"

    if not gmpy2.is_prime(start + gap_size):
        return False, "end not prime"

    assert start % 2 == 1

    tests += 2
    current = "Testing {}, {}/{} done, {} PRPs performed".format(
        gap_size, 2, gap_size, tests)

    # TODO use gmpy2.next_prime()
    #   pros: faster (~2x)
    #   cons: no status (maybe create timestamp somewhere)

    # Do something smart here like gmp-devel list
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
    if log_n > 4000 and merit < 25:
        # More trusted discoverers
        if discoverer in ("Jacobsen", "M.Jansen", "RobSmith", "Rosnthal"):
            skip_fraction = 0.97

    for k in range(2, gap_size, 2):
        if composite[k]: continue

        if skip_fraction > 0 and random.random() < skip_fraction:
            continue

        if gmpy2.is_prime(start + k):
            return False, "start + {} is prime".format(k)

        tests += 1
        current = "Testing {}, {}/{} done, {} PRPs performed".format(
            gap_size, k+1, gap_size, tests)

    return True, "Verified"


def update_all_sql(sql_insert):
    sql_lines = []
    with open(ALL_SQL_FN, "r") as f:
        sql_lines = f.readlines()

    assert 110 < len(sql_lines) < 100000, len(sql_lines)

    new_line = SQL_INSERT_PREFIX + str(sql_insert).replace(" ", "") + ";\n"

    # Find the right place in the insert section
    for index, line in enumerate(sql_lines):
        if line.startswith("INSERT INTO gaps"):
            # Get the first number from the line
            match = re.search(r"\(([0-9]+),", line)
            assert match
            gap = int(match.group(1))
            if gap >= sql_insert[0]:
                break

    # Format is wrong
    assert (100 < index < 90000), ("WEIRD INDEX", index, new_line)

    start_insert_line = SQL_INSERT_PREFIX + "(" + str(sql_insert[0])
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
    wd = os.getcwd()
    try:
        os.chdir("prime-gap-list")

        subprocess.check_output(["git", "commit", "-am", commit_msg])
        subprocess.check_output(["git", "push", "safe"])
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


def parse_start(num_str):
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
        if p > 20000:
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
        batch_num,
        gap_size, gap_start,
        ccc_type, discoverer,
        discover_date):
    if gap_size % 2 == 1:
        return False, "Odd gapsize is unlikely"
    if gap_size <= 1200:
        return False, "optimal gapsize={} has already been found".format(
            gap_size)

    gap_start = gap_start.replace(" ", "")
    if any(gap_start in line for line in new_records):
        return False, "Already added to records"

    queue.mutex.acquire()
    in_queue = any(k[1] == gap_size for k in queue.queue)
    queue.mutex.release()
    if in_queue:
        return True, "Already in queue"

    rv = list(get_db().execute(
        "SELECT merit, startprime FROM gaps WHERE gapsize = ?",
        (gap_size,)))
    assert len(rv) in (0, 1), [tuple(r) for r in rv]
    if len(rv) == 0:
        rv.append([0, 0])
    e_merit, e_startprime = rv[0]

    start_n = parse_start(gap_start)
    if start_n is None:
        err_msg = ("Can't parse gapstart={} (post on "
                   "MersenneForum if this is an error)").format(gap_start)
        return False, err_msg
    if start_n >= REALLY_LARGE:
        return False, "gapstart={} is to large at this time".format(gap_start)
    if start_n % 2 == 0:
        return False, "gapstart={} is even".format(gap_start)

    newmerit = gap_size / gmpy2.log(start_n)
    if newmerit <= e_merit + 0.005:
        return False, "Existing {} with better merit {:.3f} vs {:.4f}".format(
            e_startprime, e_merit, newmerit)

    # Pull data from form for old style line & github entry
    newmerit_fmt = round(float(newmerit), 3)
    primedigits = len(str(start_n))

    line_fmt = "{}, {}, {}, {}, {}, {}, {}".format(
        gap_size, ccc_type, newmerit_fmt, discoverer,
        discover_date.isoformat(), primedigits, gap_start)

    year = discover_date.year
    assert 2015 <= year <= 2024, year
    sql_insert = (gap_size, 0) + tuple(ccc_type) + (discoverer,
        year, newmerit_fmt, primedigits, gap_start)

    queue.put((batch_num, gap_size, start_n, line_fmt, sql_insert))

    return True, "Adding {} gapsize={} to queue, would improve merit {} to {:.3f}".format(
        short_start(start_n), gap_size,
        "{:.3f}".format(e_merit) if e_merit > 0.1 else "NONE",
        newmerit)


def possible_add_to_queue_form(form):
    return possible_add_to_queue(
        form.gapstart.data,
        form.gapsize.data,
        form.gapstart.data,
        form.ccc.data,
        form.discoverer.data,
        form.date.data)


def possible_add_to_queue_log(form):
    discoverer = form.discoverer.data
    log_data = form.logdata.data

    # How to choice this better?
    batch_num = abs(hash(discoverer) + hash(log_data))

    adds = []
    statuses = []
    for line in log_data.split("\n"):
        if len(line.strip()) == 0:
            continue

        full_log_re = (r"(\d+)\s+"
                       r"(20[12]\d-\d\d?-\d\d?)\s+([\w.]+\.?[\w.]*)\s+"
                       r"([\d.]+)\s+"
                       r"(\d+\s*\*\s*\d+#\s*/\s*\d+#?\s*\-\s*\d+)")
        match = re.search(full_log_re, line)
        if match:
            if discoverer.lower() != "file":
                statuses.append("Must set discoverer=file to use full line format")
                continue
            else:
                line_date = datetime.datetime.strptime(match.group(2), "%Y-%m-%d").date()
                added, status = possible_add_to_queue(
                    batch_num,
                    int(match.group(1)),
                    match.group(5),
                    "C?P",  # TODO describe this somewhere
                    match.group(3),
                    line_date)
                adds.append(added)
                statuses.append(status)
                continue

        log_re = r"(\d+)\s+([\d.]+)\s+(\d+\s*\*\s*\d+#\s*/\s*\d+#?\s*\-\s*\d+)"
        match = re.search(log_re, line)
        if match:
            discover_date = form.date.data
            added, status = possible_add_to_queue(
                batch_num,
                int(match.group(1)),
                match.group(3),
                "C?P",  # TODO describe this somewhere
                discoverer,
                discover_date)
            adds.append(added)
            statuses.append(status)
            continue

        statuses.append("Didn't find match in: " + line)

    return any(adds), "\n<br>\n".join(statuses)


class GapForm(FlaskForm):
    discover_valid_date = DateRange(
        min=datetime.date(2015, 1, 1),
        max=datetime.date.today() + datetime.timedelta(hours=48))

    type_choices = (
        ("C?P", "(C??) PRP on endpoints"),
        ("C?C", "(C?P) certified endpoints")
    )

    gapsize = IntegerField(
        "Gap Size",
        description="Gap size",
        render_kw={"style": "width:80px"},
        validators=[DataRequired()])

    ccc = SelectField(
        "CCC",
        choices=type_choices,
        description="C?? or C?P",
        validators=[DataRequired()])

    discoverer = StringField(
        "Discoverer",
        render_kw={"size": 12},
        validators=[DataRequired()])

    date = DateField(
        "Date",
        validators=[DataRequired(), discover_valid_date])

    gapstart = StringField(
        "Gapstart",
        description="Gap start",
        render_kw={"size": 30},
        validators=[DataRequired()])

    submit = SubmitField("Add")


class GapLogForm(FlaskForm):
    discover_valid_date = DateRange(
        min=datetime.date(2015, 1, 1),
        max=datetime.date.today() + datetime.timedelta(hours=48))

    discoverer = StringField(
        "Discoverer",
        render_kw={"size": 12},
        validators=[DataRequired()])

    date = DateField(
        "Date",
        validators=[DataRequired(), discover_valid_date])

    logdata = TextAreaField(
        "LogData",
        description=(
            "206048  20.785  100017163 * 10007#/30030 -138324 to +67724\n"
            "or\n"
            "22558   23.779  104304433*977#/7#-15234"
            "or\n"
            "2074 2019-09-25 M.Jansen 28.311600 3513398427*71#/30030 - 1532\n"
        ),
        render_kw={"cols": 70, "rows": 10},
        validators=[DataRequired()])

    submit = SubmitField("Add")


@app.route("/", methods=("GET", "POST"))
def controller():
    global queue

    formA = GapForm()
    formB = GapLogForm()
    which_form = request.args.get("form")

    status = ""
    added = False
    if which_form is not None and queue.qsize() > 30:
        return "Long queue try again later"

    if   which_form == "A" and formA.validate_on_submit():
        added, status = possible_add_to_queue_form(formA)
    elif which_form == "B" and formB.validate_on_submit():
        added, status = possible_add_to_queue_log(formB)

    queued = queue.qsize()
    queue_data = [k[3] for k in queue.queue]

    if added:
        # Clear both errors
        formA.errors.clear()
        formB.errors.clear()

    return render_template(
        "record-check.html",
        formA=formA,
        formB=formB,
        added=added,
        status=status,
        queued=queued,
        queue=queue_data,
    )


@app.route("/status")
def status():
    global new_records, recent, queue, worker

    queue_data = [k[3] for k in queue.queue]
    queued = len(queue_data)
    return render_template(
        "status.html",
        running=worker.is_alive(),
        new_records=new_records,
        recent=recent[-50:],
        queue=queue_data,
        queued=queued)


@app.route("/validate-gap")
def stream():
    def gap_status_stream():
        global queue, current
        if not queue.qsize():
            # avoid Done print statement
            return

        for i in range(3600):
            queued = queue.qsize()
            if not queued:
                break
            queued = queue.qsize()
            state = "Queue {}: {}".format(queued, current)
            yield "data: " + state + "\n\n"
            time.sleep(5)

        yield "data: Done (refresh to see recent status)\n\n"

    return Response(gap_status_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    # Create background gap_worker
    worker = threading.Thread(target=gap_worker)
    worker.start()

    app.run(
        host="0.0.0.0",
        # host = "::",
        port=5090,
        # debug=False,
        debug=True,
    )
