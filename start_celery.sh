#!/usr/bin/env bash

celery -A AudioProcessCeleryWorker worker -l WARNING -c 1 --pool=solo -Q GPTSoVits -E