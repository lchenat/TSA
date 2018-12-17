pattern='sample'

search:
	grep --include=*.py -rnw '.' -e $(pattern)
