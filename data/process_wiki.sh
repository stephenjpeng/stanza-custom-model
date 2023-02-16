#!/bin/bash

for f in *.conllu; do
	awk '{print $2 "\t" $3}' "$f" > wiki_"$f"
done
