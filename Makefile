format:
	# remove unused imports
	autoflake --in-place --remove-all-unused-imports \
		--ignore-init-module-imports --ignore-pass-after-docstring -r *
	# sort imports
	isort . --skip lib
	# format with YAPF
	yapf -irp . --exclude lib