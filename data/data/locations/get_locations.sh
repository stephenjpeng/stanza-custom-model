#!/bin/bash

# Chunk into continents to get around row limit
continents=( AF AS EU NA OC SA AN )

for c in "${continents[@]}"
do
	for (( i=1; i<=5; i++ ))
	do
		 curl "api.geonames.org/search?continentCode=$c&name=playa+beach+coast+shore+waterfront&style=FULL&username=stephenjpeng&operator=OR&maxRows=1000&startRow=$((i*1000))" > "$c$i"".xml"
	done
done
