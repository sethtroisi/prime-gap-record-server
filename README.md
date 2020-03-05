## Prime Records

A frontend to submit records to [prime-gap-list](https://github.com/primegap-list-project/prime-gap-list)

Performs validation then submits record.

```shell
# Setup
git submodule init
sqlite3 gaps.db < prime-gap-list/allgaps.sql
cd prime-gap-list
git remote add safe git@github.com:sethtroisi/prime-gap-list.git
git config --global user.name
git config --global user.email
# Make sure ssh exists
```
