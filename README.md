## Prime Records

A frontend to submit records to [prime-gap-list](https://github.com/primegap-list-project/prime-gap-list)

Performs validation then submits record.

```shell
pip install gmpy2 primegapverify

# Setup
git clone https://github.com/sethtroisi/prime-gap-record-server
cd prime-gap-record-server

git clone https://github.com/primegap-list-project/prime-gap-list.git
cd prime-gap-list
sqlite3 gaps.db < allgaps.sql
sudo chmod 664 gaps.db
cd ..

python flask_serve.py
```
A local server should now be running at http://localhost:5090

For the official server a few extra steps are needed
```
git remote add upstream git@github.com:primegap-list-project/prime-gap-list.git
git remote remove origin
git fetch
git config --global user.name
git config --global user.email
# Make sure ssh works

# Because of complicated configs
truncate -s0 gaps.db; sqlite3 gaps.db < allgaps.sql
touch records.txt submissions.txt
sudo chown -R www-data records.txt submissions.txt prime-gap-list/
```

### TODO

* [ ] Write failed verifications to error.txt
* [x] Handle multiple records for same gap in same batch
* [ ] Test permissions page
  * [ ] write to records
  * [ ] verify git setup
  * [x] can write to db
* [ ] Create PR for new contributor
* [ ] Push for regular contributor
* [x] C?? for very large gaps.
* [x] Change to multiprocess
* [x] Figure out out how to correctly compute merit for existing records.
* [x] Bundle same discoverer into batches
* [x] Replace line in allgaps.sql if record already exists
* [x] Scroll areas in status
