# Minimal shim to satisfy Pyubiomes' unconditional nether import.
# We do not support nether features in this project; any call will raise.

class NetherBiomes:
	NetherWastes = 0

class NetherGen:  # placeholder type used by the bindings
	pass


def create_new_nether(seed):
	return NetherGen()


def get_biome(*args, **kwargs):
	raise NotImplementedError("Nether biome queries are not supported in this project.")


def get_biome_structure(*args, **kwargs):
	raise NotImplementedError("Nether structure queries are not supported in this project.")


def get_biome_decorator(*args, **kwargs):
	raise NotImplementedError("Nether decorator queries are not supported in this project.")


def delete(*args, **kwargs):
	return None
