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
from flask import Flask, g, render_template, Response
from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, SubmitField
from wtforms.validators import DataRequired
from wtforms_components import IntegerField, DateField, DateRange


print("Setting up prime-records controller")

app = Flask(__name__)

SECRET_KEY = hex(random.randint(0, 2**32))
app.config['SECRET_KEY'] = SECRET_KEY

RECORDS_FN = "records.txt"
ALL_SQL_FN = "prime-gap-list/allgaps.sql"

SQL_INSERT_PREFIX = "INSERT INTO gaps VALUES"
assert os.path.isfile(ALL_SQL_FN), "git init submodule first"

REALLY_LARGE = 10 ** 10000


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

            # Write to temp file
            with open(RECORDS_FN, 'a') as f:
                f.write(line_fmt + "\n")

            # Write to allgaps.sql (sorted), kinda slow
            update_all_sql(sql_insert)

            # Write to gaps.db
            with open_db() as db:
                db.execute(SQL_INSERT_PREFIX + str(sql_insert))
                db.commit()


def test_one(gap_size, start, line_fmt, sql_insert):
    global current

    tests = 0
    current = (gap_size, 0, tests)

    if not gmpy2.is_prime(start):
        return False, "start not prime"

    tests += 1

    if not gmpy2.is_prime(start + gap_size):
        return False, "end not prime"

    tests += 1
    current = (gap_size, 0, tests)

    composite = [False for i in range(gap_size+1)]
    sieve_primes = 1000000
    primes = [True for i in range(sieve_primes+1)]
    for p in range(2, sieve_primes):
        if not primes[p]: continue
        # Sieve other primes
        for m in range(p*p, sieve_primes+1, p):
            primes[m] = False

        # Remove any numbers in the interval
        first = -start % p
        for m in range(first, gap_size+1, p):
            assert (start + m) % p == 0
            composite[m] = True

    assert composite[0] is False and composite[-1] is False

    # Do something better here with sieving small primes in the whole interval
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

    line = SQL_INSERT_PREFIX + str(sql_insert).replace(' ', '') + "\n"

    # Find the right place in the insert section
    for index, line in enumerate(sql_lines):
        if line.startswith("INSERT INTO gaps"):
            # get the first number from the line
            match = re.search(r"\(([0-9]+),", line)
            assert match
            gap = int(match.group(1))
            if gap > sql_insert[0]:
                break

    # We have the format wrong
    if not (100 < index < 90000):
        print("WEIRD INDEX", index)
        print(sql_lines[index])

    sql_lines.insert(index, line)

    with open(ALL_SQL_FN, 'w') as f:
        for line in sql_lines:
            f.write(line)

    wd = os.getcwd()
    try:
        os.chdir("prime-gap-list")

        commit_msg = "New Record {} merit={} uploaded by {}".format(
            sql_insert[0], sql_insert[7], sql_insert[-1])

        subprocess.check_call(['git', 'commit', '-am', commit_msg])
        #subprocess.check_call(['git', 'push', 'safe'])
    except Exception as e:
        print("Error!", e)
    os.chdir(wd)


def open_db():
    return sqlite3.connect("gaps.db")


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


def parse_start(gapstart_str):
    if '(' in gapstart_str or ')' in gapstart_str:
        return None
    if len(gapstart_str) > 10000:
        return None

    # Remove whitespace from string
    gapstart_str = re.sub(r'\s', '', gapstart_str)

    for generation in range(100):
        # Parse Exponents
        match = re.search(r'([0-9]+)\^([0-9]+)', gapstart_str)
        if match:
            base, exp = map(int, match.groups())
            if gmpy2.log(base) * exp > 10100:
                return REALLY_LARGE
            result = base ** exp
            gapstart_str = gapstart_str.replace(match.group(0), str(result))
            continue

        match = re.search(r'([0-9]+)#', gapstart_str)
        if match:
            p = int(match.group(1))
            if p > 20000:
                return REALLY_LARGE
            result = gmpy2.primorial(p)
            gapstart_str = gapstart_str.replace(match.group(0), str(result))
            continue

        match = re.search(r'([0-9]+)([/*])([0-9]+)', gapstart_str)
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
            gapstart_str = gapstart_str.replace(match.group(0), str(result))
            continue

        match = re.search(r'([0-9]+)([+-])([0-9]+)', gapstart_str)
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
            gapstart_str = gapstart_str.replace(match.group(0), str(result))
            continue

        if re.search(r'^[0-9]+$', gapstart_str):
            return int(gapstart_str)

    return None


def possible_add_to_queue(form):
    gap_size = int(form.gapsize.data)
    gap_start = form.gapstart.data.replace(' ', '')

    if gap_size % 2 == 1:
        return False, "Odd gapsize is unlikely"
    if gap_size <= 1200:
        return False, "optimal gapsize={} has already been found".format(
            gap_size)

#    if any(gap_start in line for line in new_records):
#        return False, "Already added to records"

    if any(k[0] == gap_size for k in queue.queue):
        return True, "Already in queue"

    db = get_db()
    rv = list(db.execute(
        "SELECT merit, startprime FROM gaps WHERE gapsize = ?",
        (gap_size,)))
    assert len(rv) in (0, 1)
    if len(rv):
        e_merit, e_startprime = rv[0]
    else:
        e_merit, e_startprime = 0, 0

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
    if newmerit < e_merit:
        return False, "Existing {} with better merit {:.3f} vs {:.4f}".format(
            e_startprime, e_merit, newmerit)

    # Pull data from form for old style line & github entry
    newmerit_fmt = round(float(newmerit), 3)
    primedigits = len(str(start_n))

    line_fmt = "{}, {}, {}, {}, {}, {}, {}".format(
        gap_size,
        form.ccc.data,
        newmerit_fmt,
        form.discoverer.data,
        form.date.data.isoformat(),
        primedigits,
        gap_start)

    year = form.date.data.year
    assert 2015 <= year <= 2024, year
    sql_insert = (gap_size, 0) + tuple(form.ccc.data) + (
        form.discoverer.data, year, newmerit_fmt, primedigits, gap_start)

    queue.put((gap_size, start_n, line_fmt, sql_insert))
    return True, "Adding {} gapsize={} to queue, would improve merit {:.3f} to {:.3f}".format(
        short_start(start_n), gap_size, e_merit, newmerit)


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
        render_kw={'size': 15},
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


@app.route('/', methods=('GET', 'POST'))
def controller():
    global queue
    form = GapForm()

    status = ""
    added = False
    if form.validate_on_submit():
        if queue.qsize() > 10:
            return "Long queue try again later"
        added, status = possible_add_to_queue(form)

    queued = queue.qsize()
    queue_data = [k[2] for k in queue.queue]

    return render_template(
        'record-check.html',
        form=form,
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
