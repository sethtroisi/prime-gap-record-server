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
app.config['SECRET_KEY'] = SECRET_KEY

RECORDS_FN = "records.txt"
ALL_SQL_FN = "prime-gap-list/allgaps.sql"
GAPS_DB_FN = "gaps.db"

SQL_INSERT_PREFIX = "INSERT INTO gaps VALUES"
assert os.path.isfile(ALL_SQL_FN), "git init submodule first"

REALLY_LARGE = 10 ** 10000
SIEVE_PRIMES = 1000000


# Globals for exchanging queue info with background thread
new_records = []
if os.path.isfile(RECORDS_FN):
    with open(RECORDS_FN, 'r') as f:
        new_records = f.readlines()
recent = []
queue = Queue()
current = None
worker = None


def gap_worker():
    global new_records, recent, queue, current

    print("Gap Worker running")

    while True:
        current = None
        # Blocks without busy wait
        queue.not_empty.acquire()
        queue.not_empty.wait()
        queue_0 = queue.queue[0]
        queue.not_empty.release()

        gap_size, start, line_fmt, sql_insert = queue_0
        human = sql_insert[-1]

        start_t = time.time()
        success, status = test_one(gap_size, start, line_fmt, sql_insert)
        end_t = time.time()

        popped = queue.get()
        assert popped == queue_0

        if not success:
            recent.append((gap_size, human, status))
        else:
            status = "Verified! ({:.1f}s)".format(end_t - start_t)
            recent.append((gap_size, human, status))
            new_records.append(line_fmt)

            print(line_fmt)
            print(sql_insert)

            # Write to record file
            with open(RECORDS_FN, 'a') as f:
                f.write(line_fmt + "\n")

            # Write to allgaps.sql (sorted), kinda slow
            update_all_sql(sql_insert)

            # Write to gaps.db
            with open_db() as db:
                # Delete any existing gap
                db.execute("DELETE FROM gaps WHERE gapsize = ?", (gap_size,))

                # Insert new gap into db
                db.execute(SQL_INSERT_PREFIX + str(sql_insert))
                db.commit()


def test_one(gap_size, start, line_fmt, sql_insert):
    global current

    tests = 0
    current = (gap_size, 0, tests)

    if not gmpy2.is_prime(start):
        return False, "start not prime"

    if not gmpy2.is_prime(start + gap_size):
        return False, "end not prime"

    tests += 2
    current = (gap_size, 0, tests)

    composite = [False for i in range(gap_size+1)]
    primes = [True for i in range(SIEVE_PRIMES+1)]
    for p in range(2, SIEVE_PRIMES):
        if not primes[p]: continue
        # Sieve other primes
        for m in range(p*p, SIEVE_PRIMES+1, p):
            primes[m] = False

        # Remove any numbers in the interval
        first = -start % p
        for m in range(first, gap_size+1, p):
            # assert (start + m) % p == 0
            composite[m] = True

    assert composite[0] is False and composite[-1] is False

    for k in range(2, gap_size, 2):
        if composite[k]: continue

        if gmpy2.is_prime(start + k):
            return False, "start + {} is prime".format(k)

        tests += 1
        current = (gap_size, k, tests)

    return True, "Verified"


def update_all_sql(sql_insert):
    sql_lines = []
    with open(ALL_SQL_FN, 'r') as f:
        sql_lines = f.readlines()

    new_line = SQL_INSERT_PREFIX + str(sql_insert).replace(' ', '') + "\n"

    # Find the right place in the insert section
    for index, line in enumerate(sql_lines):
        if line.startswith("INSERT INTO gaps"):
            # Get the first number from the line
            match = re.search(r"\(([0-9]+),", line)
            assert match
            gap = int(match.group(1))
            if gap > sql_insert[0]:
                break

    # We have the format wrong
    if not (100 < index < 90000):
        print("WEIRD INDEX", index)
        print(sql_lines[index])

    sql_lines.insert(index, new_line)

    with open(ALL_SQL_FN, 'w') as f:
        for line in sql_lines:
            f.write(line)

    wd = os.getcwd()
    try:
        os.chdir("prime-gap-list")

        commit_msg = "New Record {} merit={} uploaded by {}".format(
            sql_insert[0], sql_insert[7], sql_insert[5])

        subprocess.check_call(['git', 'commit', '-am', commit_msg])
        subprocess.check_call(['git', 'push', 'safe'])
    except Exception as e:
        print("Error!", e)
    os.chdir(wd)


def open_db():
    return sqlite3.connect(GAPS_DB_FN)


def get_db():
    db = getattr(g, '_database', None)
    # Setup
    if db is None:
        db = g._database = open_db()
    db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def short_start(n):
    str_n = str(n)
    if len(str_n) < 20:
        return str_n
    return "{}...{}<{}>".format(str_n[:3], str_n[-3:], len(str_n))


def parse_start(num_str):
    if '(' in num_str or ')' in num_str:
        return None
    if len(num_str) > 10000:
        return None

    # Remove whitespace from string
    num_str = re.sub(r'\s', '', num_str)

    for generation in range(20):
        # Parse Exponents
        if '^' in num_str:
            match = re.search(r'(\d+)\^(\d+)', num_str)
            if match:
                base, exp = map(int, match.groups())
                if gmpy2.log(base) * exp > 10100:
                    return REALLY_LARGE
                result = base ** exp
                num_str = num_str.replace(match.group(0), str(result))
                continue

        if '#' in num_str:
            match = re.search(r'(\d+)#', num_str)
            if match:
                p = int(match.group(1))
                if p > 20000:
                    return REALLY_LARGE
                result = gmpy2.primorial(p)
                num_str = num_str.replace(match.group(0), str(result))
                continue

        if '*' in num_str or '/' in num_str:
            match = re.search(r'(\d+)([/*])(\d+)', num_str)
            if match:
                a, sign, b = match.groups()
                a_n = int(a)
                b_n = int(b)
                if sign == '*':
                    if len(a) + len(b) > 10100:
                        return REALLY_LARGE
                    result = a_n * b_n
                elif sign == '/':
                    if b_n == 0 or a_n % b_n != 0:
                        return None
                    result = a_n // b_n
                else:
                    return None
                num_str = num_str.replace(match.group(0), str(result))
                continue

        if '+' in num_str or '-' in num_str:
            match = re.search(r'([0-9]+)([+-])([0-9]+)', num_str)
            if match:
                a, sign, b = match.groups()
                a_n = int(a)
                b_n = int(b)
                if sign == '+':
                    result = a_n + b_n
                elif sign == '-':
                    result = a_n - b_n
                else:
                    return None
                num_str = num_str.replace(match.group(0), str(result))
                continue

        if re.search(r'^[0-9]+$', num_str):
            return int(num_str)

    return None

def possible_add_to_queue(
        gap_size, gap_start,
        ccc_type, discoverer,
        discover_date):
    if gap_size % 2 == 1:
        return False, "Odd gapsize is unlikely"
    if gap_size <= 1200:
        return False, "optimal gapsize={} has already been found".format(
            gap_size)

    gap_start = gap_start.replace(' ', '')
    if any(gap_start in line for line in new_records):
        return False, "Already added to records"
    if any(k[0] == gap_size for k in queue.queue):
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
        return False, "Existing {} with better (or close) merit {:.3f} vs {:.4f}".format(
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

    queue.put((gap_size, start_n, line_fmt, sql_insert))
    return True, "Adding {} gapsize={} to queue, would improve merit {:.3f} to {:.3f}".format(
        short_start(start_n), gap_size, e_merit, newmerit)


def possible_add_to_queue_form(form):
    return possible_add_to_queue(
        form.gapsize.data, # TODO int(...) ?
        form.gapstart.data,
        form.ccc.data,
        form.discoverer.data,
        form.date.data)

def possible_add_to_queue_log(form):
    discoverer = form.discoverer.data
    discover_date = form.date.data
    log_data = form.logdata.data

    adds = []
    statuses = []
    for line in log_data.split("\n"):
        log_re = r'(\d+)\s+([0-9.]+)\s+(\d+\s*\*\s*\d+#\s*/\s*\d+#?\s*\-\s*\d+)'
        match = re.search(log_re, line)
        if match:
            added, status = possible_add_to_queue(
                int(match.group(1)),
                match.group(3),
                "C?P", # TODO describe this somewhere
                discoverer,
                discover_date)
            adds.append(added)
            statuses.append(status)
        else:
            statuses.append("Didn't find match in: " + line)

    return any(adds), "\n<br>\n".join(statuses)


class GapForm(FlaskForm):
    discover_valid_date = DateRange(
        min=datetime.date(2015, 1, 1),
        max=datetime.date.today() + datetime.timedelta(hours=48))

    type_choices = (
        ('C?P', '(C??) PRP on endpoints'),
        ('C?C', '(C?P) certified endpoints')
    )

    gapsize = IntegerField(
        'Gap Size',
        description='Gap size',
        render_kw={'style': 'width:80px'},
        validators=[DataRequired()])

    ccc = SelectField(
        'CCC',
        choices=type_choices,
        description='C?? or C?P',
        validators=[DataRequired()])

    discoverer = StringField(
        'Discoverer',
        render_kw={'size': 12},
        validators=[DataRequired()])

    date = DateField(
        'Date',
        validators=[DataRequired(), discover_valid_date])

    gapstart = StringField(
        'Gapstart',
        description='Gap start',
        render_kw={'size': 30},
        validators=[DataRequired()])

    submit = SubmitField('Add')


class GapLogForm(FlaskForm):
    discover_valid_date = DateRange(
        min=datetime.date(2015, 1, 1),
        max=datetime.date.today() + datetime.timedelta(hours=48))

    discoverer = StringField(
        'Discoverer',
        render_kw={'size': 12},
        validators=[DataRequired()])

    date = DateField(
        'Date',
        validators=[DataRequired(), discover_valid_date])

    logdata = TextAreaField(
        'LogData',
        description=(
            '206048  20.7850  100017163 * 10007#/30030 -138324 to +67724'),
        render_kw={'cols': 70, 'rows': 10},
        validators=[DataRequired()])

    submit = SubmitField('Add')


@app.route('/', methods=('GET', 'POST'))
def controller():
    global queue

    formA = GapForm()
    formB = GapLogForm()
    which_form = request.args.get('form')

    status = ""
    added = False
    if which_form is not None and queue.qsize() > 30:
        return "Long queue try again later"

    if   which_form == 'A' and formA.validate_on_submit():
        added, status = possible_add_to_queue_form(formA)
    elif which_form == 'B' and formB.validate_on_submit():
        added, status = possible_add_to_queue_log(formB)

    queued = queue.qsize()
    queue_data = [k[2] for k in queue.queue]

    if added:
        # Clear both errors
        formA.errors.clear()
        formB.errors.clear()

    return render_template(
        'record-check.html',
        formA=formA,
        formB=formB,
        added=added,
        status=status,
        queued=queued,
        queue=queue_data,
    )

@app.route('/status')
def status():
    global new_records, recent, queue, current, worker

    queue_data = [k[2] for k in queue.queue]
    queued = len(queue_data)
    return render_template(
        'status.html',
        running=worker.is_alive(),
        new_records=new_records,
        recent=recent,
        queue=queue_data,
        current=current,
        queued=queued)


@app.route('/validate-gap')
def stream():
    def gap_status_stream():
        global queue

        queued = queue.qsize()
        while queued:
            queued = queue.qsize()
            state = "Queue {}: Currently Testing gap={}:".format(
                queued, current)
            yield "data: " + state + "\n\n"
            time.sleep(1)
        yield 'data: Done (refresh to see recent status)'

    return Response(gap_status_stream(), mimetype='text/event-stream')


if __name__ == "__main__":
    # Create background gap_worker
    worker = threading.Thread(target=gap_worker)
    worker.start()

    app.run(
        host='0.0.0.0',
        #host = '::',
        port=5090,
        #debug=False,
        debug=True,
    )
