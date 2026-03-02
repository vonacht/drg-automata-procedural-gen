uv run main.py -k SMALL_SLOPE SMALL_FLAT SMALL_VERTICAL MEDIUM_FLAT MEDIUM_SLOPE MEDIUM_VERTICAL BIG_FLAT BIG_SLOPE BIG_VERTICAL -n 20 -c settings/settings_mixed.json
uv run main.py -k ESCORT_MEDIUM ESCORT_SMALL END -n 25 -c settings/settings_mixed.json
uv run main.py -t -m 
uv run main.py -r -m
