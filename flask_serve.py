import datetime
import random
import re
import sqlite3
import threading
import time
from queue import Queue

from flask import Flask, g, render_template, Response, request
from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, SubmitField
from wtforms.validators import DataRequired, ValidationError
from wtforms_components import IntegerField, DateField, DateRange

import gmpy2

print ("Setting up prime-records controller")

app = Flask(__name__)

SECRET_KEY = hex(random.randint(0, 2**32))
app.config['SECRET_KEY'] = SECRET_KEY

# Save to disk and reload (or just in prime-gaps.db)
recent = []
queue = Queue()
current = None


def shortStart(n):
    str_n = str(n)
    if len(str_n) < 20:
        return str_n
    return "{}...{}<{}>".format(str_n[:3], str_n[-3:], len(str_n))


def gapWorker():
    global recent, queue, current

    print ("Gap Worker running")

    while True:
        # Blocks without busy wait
        current = None
        gapSize, start, lineFmt, sqlInsert = queue.get()
        start_t = time.time()

        abbr = shortStart(start)
        print ("Testing gapsize={}, start={}".format(gapSize, abbr))

        tests = 0
        current = (gapSize, 0, tests)

        if not gmpy2.is_prime(start):
            recent.append((gapSize, abbr, "start not prime"))
            continue
        tests += 1
        current = (gapSize, 0, tests)

        if not gmpy2.is_prime(start + gapSize):
            recent.append((gapSize, abbr, "end not prime"))
            continue
        tests += 1
        current = (gapSize, 0, tests)

        composite = [False for i in range(gapSize+1)]
        sieve_primes = 1000000
        primes = [True for i in range(sieve_primes+1)]
        for p in range(2, sieve_primes):
            if not primes[p]: continue
            # Sieve other primes
            for m in range(p*p, sieve_primes+1, p):
                primes[m] = False

            # Remove any numbers in the interval
            first = -start % p
            for m in range(first, gapSize+1, p):
                assert (start + m) % p == 0
                composite[m] = True

        assert composite[0] == False and composite[-1] == False

        # Do something better here with sieving small primes in the whole interval
        for k in range(2, gapSize, 2):
            if composite[k]: continue

            if gmpy2.is_prime(start + k):
                recent.append((gapSize, abbr, "start + {} is prime".format(k)))
                break
            tests += 1
            current = (gapSize, k, tests)
        else:
            end_t = time.time()
            recent.append((gapSize, abbr, "Good! ({:.1f}s)".format(end_t - start_t)))

            # TODO create git submit ...
            print (lineFmt)
            print (sqlInsert)


class GapForm(FlaskForm):
    discover_valid_date = DateRange(
        min=datetime.date(2015, 1, 1),
        max=datetime.date.today() + datetime.timedelta(hours=48))

    typeChoices = (('C?P', '(C??) PRP on endpoints'),
                   ('C?C', '(C?P) certified endpoints'))

    gapsize     = IntegerField('Gap Size',
        description='Gap size',
        render_kw={'style': 'width:80px'},
        validators=[DataRequired()])

    ccc         = SelectField('CCC',
        choices=typeChoices,
        description='C?? or C?P',
        validators=[DataRequired()])

    discoverer  = StringField('Discoverer',
        render_kw={'size': 15},
        validators=[DataRequired()])

    date        = DateField('Date',
        validators=[DataRequired(), discover_valid_date])

    gapstart    = StringField('Gapstart',
        description='Gap start',
        render_kw={'size':30},
        validators=[DataRequired()])

    submit = SubmitField('Add')


def get_db():
    db = getattr(g, '_database', None)
    # Setup
    if db is None:
        db = g._database = sqlite3.connect("gaps.db")
    db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


REALLY_LARGE = 10 ** 10000
def parseStart(gapstart_str):
    n = 1
    if '(' in gapstart_str or ')' in gapstart_str:
        return None
    if len(gapstart_str) > 10000:
        return None

    # Remove whitespace from string
    gapstart_str = re.sub(r'\s', '', gapstart_str)

    for generation in range(100):
        # Parse Exponents
        expMatch = re.search(r'([0-9]+)\^([0-9]+)', gapstart_str)
        if expMatch:
            base, exp = map(int, expMatch.groups())
            if gmpy.log(base) * exp > 10100:
                return REALLY_LARGE
            result = base ** exp
            gapstart_str = gapstart_str.replace(expMatch.group(0), str(result))
            continue

        primorialMatch = re.search(r'([0-9]+)#', gapstart_str)
        if primorialMatch:
            p = int(primorialMatch.group(1))
            if p > 20000:
                return REALLY_LARGE
            result = gmpy2.primorial(p)
            gapstart_str = gapstart_str.replace(primorialMatch.group(0), str(result))
            continue

        mdMatch = re.search(r'([0-9]+)([/*])([0-9]+)', gapstart_str)
        if mdMatch:
            a, sign, b = mdMatch.groups()
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
            gapstart_str = gapstart_str.replace(mdMatch.group(0), str(result))
            continue

        asMatch = re.search(r'([0-9]+)([+-])([0-9]+)', gapstart_str)
        if asMatch:
            a, sign, b = asMatch.groups()
            a_n = int(a)
            b_n = int(b)
            if sign == '+':
                result = a_n + b_n
            elif sign == '-':
                result = a_n - b_n
            else:
                return None
            gapstart_str = gapstart_str.replace(asMatch.group(0), str(result))
            continue

        nMatch = re.search(r'^[0-9]+$', gapstart_str)
        if nMatch:
            return int(gapstart_str)

    return None


def possibleAddToQueue(form):
    gapSize = int(form.gapsize.data)
    gapStart = form.gapstart.data

    if gapSize % 2 == 1:
        return False, "Odd gapsize is unlikely"
    if gapSize <= 1200:
        return False, "optimal gapsize={} has already been found".format(gapSize)

    # TODO also check recent

    if any(k[0] == gapSize for k in queue.queue):
        return True, "Already in queue"

    db = get_db()
    rv = list(db.execute(
        "SELECT merit, primedigits, startprime FROM gaps WHERE gapsize = ?",
        (gapSize,)))
    assert len(rv) in (0, 1)
    if len(rv):
        merit, primedigits, startprime = rv[0]
    else:
        merit, primedigits, startprime = 0, 10 ** 6, 0

    start_n = parseStart(gapStart)
    if start_n is None:
        return False, "Can't parse gapstart={} (post on MersenneForum if this is an error)".format(
            gapStart)
    if start_n >= REALLY_LARGE:
        return False, "gapstart={} is to large at this time".format(gapStart)

    if start_n % 2 == 0:
        return False, "gapstart={} is even".format(gapStart)

    newmerit = gapSize / gmpy2.log(start_n)
    if newmerit < merit:
        return False, "Existing gap after {} with merit {:.3f} > {:.4f}".format(
            startprime, merit, newmerit)


    # Pull data from form for old style line & github entry
    newmerit_fmt = "{:.3f}".format(newmerit)
    primedigits = len(str(start_n))

    lineFmt = "{},{},{},{},{},{},{}".format(
        gapSize,
        form.ccc.data,
        newmerit_fmt,
        form.discoverer.data,
        form.date.data.isoformat(),
        primedigits,
        gapStart)

    year = form.date.data.year
    assert 2015 <= year <= 2024, year
    sqlInsert = (gapSize, False) + tuple(form.ccc.data) + (
        form.discoverer.data, year, newmerit_fmt, primedigits, gapStart)


    queue.put((gapSize, start_n, lineFmt, sqlInsert))
    return True, "Adding {} gapsize={} to queue, would improve merit {:.3f} to {:.3f}".format(
        shortStart(start_n), gapSize, merit, newmerit)


@app.route('/', methods=('GET', 'POST'))
def controller():
    global recent, queue
    form = GapForm()
    queued = queue.qsize() + (current is not None)

    added = False
    status = "Current Queue: {} check{}".format(queued, "s" if queued > 0 else "")

    if form.validate_on_submit():
        if queue.qsize() > 10:
            return "Long queue try again later"
        added, status = possibleAddToQueue(form)

    return render_template('record-check.html',
        form=form,
        added=added,
        status=status,
        queued=queued,
    )


@app.route('/validate-gap')
def validate():
    def gapStatusStream():
        global queue

        queued = queue.qsize() + (current is not None)
        while queued:
            queued = queue.qsize() + (current is not None)
            state = "Queue {}: Currently Testing gap={}:".format(
                queued, current)
            yield "data: " + state + "\n\n"
            time.sleep(1)
        yield 'data: Done (refresh to see recent status)'
        print("Statuing done2")

    return Response(gapStatusStream(), mimetype='text/event-stream')


if __name__ == "__main__":
    # Create background gapWorker
    background = threading.Thread(target=gapWorker)
    background.start()

    app.run(
        host='0.0.0.0',
        #host = '::',
        port = 5090,
#        debug = False,
        debug = True,
    )
