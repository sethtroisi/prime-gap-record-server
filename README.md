## Prime Records

A frontend to submit records to [prime-gap-list](https://github.com/primegap-list-project/prime-gap-list)

Performs validation then submits record.

```shell
# Setup
git clone --branch server https://github.com/primegap-list-project/prime-gap-list.git
sqlite3 gaps.db < prime-gap-list/allgaps.sql
cd prime-gap-list
git remote add upstream git@github.com:primegap-list-project/prime-gap-list.git
git remote remove origin
git fetch
git config --global user.name
git config --global user.email
# Make sure ssh exists

# Because of complicated configs
truncate -s0 gaps.db; sqlite3 gaps.db < prime-gap-list/allgaps.sql
touch records.txt submissions.txt
sudo chown -R www-data records.txt submissions.txt gaps.db prime-gap-list/

```

### TODO

* [ ] C?? for very large gaps.
* [ ] Create PR for new contributor
* [ ] Push for regular contributor
* [x] Change to multiprocess
* [x] Figure out out how to correctly compute merit for existing records.
* [x] Bundle same discoverer into batches
* [x] Replace line in allgaps.sql if record already exists
* [x] Scroll areas in status
