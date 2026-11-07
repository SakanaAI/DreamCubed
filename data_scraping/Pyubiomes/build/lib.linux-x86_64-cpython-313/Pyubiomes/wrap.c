// check the biome at a block position
//uwu
#include "overworld.c"
#include <stdio.h>
#include <Python.h>
#include <string.h>
static PyObject* pyBiomeAtPos(PyObject* self, PyObject* args)
{
	int biome, xpos, zpos, version;
	long seed; 
    
    if(!PyArg_ParseTuple(args, "iliii", &biome, &seed, &xpos, &zpos, &version))
        return NULL;
	
    return Py_BuildValue("i", cBiomeAtPos(biome, seed, xpos, zpos, version));

}
PyDoc_STRVAR(biomeAtPos_doc, "biomeAtPos(biome, seed, xpos, zpos, version)->Bool\n\n\
Returns True if the [biome] biome is at position (xpos, zpos). Otherwise returns False");

static PyObject *pyBiomesInArea(PyObject *self, PyObject *args)
{
	
    PyObject* biomes;
    int biomes_array_length;
	int64_t seed; 
	int x1, z1, x2, z2, version;

	if (!PyArg_ParseTuple(args, "Oliiiii", &biomes, &seed, &x1, &z1, &x2, &z2, &version)) return NULL;

	biomes_array_length = (int)PyObject_Length(biomes);
	int biomes_array[biomes_array_length];
    
    if (biomes_array_length < 0)
        return NULL;
        
    for (int index = 0; index < biomes_array_length; index++) {
        PyObject *item;
        item = PyList_GetItem(biomes, index);
        if (!PyLong_Check(item)){biomes_array[index] = 0;}

        biomes_array[index] = (int)PyLong_AsLong(item);
    }

    return Py_BuildValue("O", PyBool_FromLong((long)cBiomesInArea(biomes_array, biomes_array_length, seed, x1, z1, x2, z2, version)));
}

PyDoc_STRVAR(biomesInArea_doc, "biomesInArea(biomes, seed, x1, z1, x2, z2, version)->Bool\n\n\
Returns True if all of the [biomes] are in the rectangular area of width [w] and height [h] with the most negative corner being at (x1,z1). Otherwise returns False");
static PyObject* pyGetSpawn(PyObject* self, PyObject* args){

	int version;
	int64_t seed;

	if (!PyArg_ParseTuple(args, "li", &seed, &version)) return NULL;
	
	Pos p=cGetSpawn(seed, version);
	
	return Py_BuildValue("(ii)", p.x, p.z );
}
PyDoc_STRVAR(getSpawn_doc, "getSpawn(seed, version)->Tuple\n\n\
 Returns a tuple containing the worldspawn for [seed] in minecraft version [version]");


static PyObject* pyFindStructure(PyObject* self, PyObject* args)
{
	int structType, x1, z1, x2, z2, version;

    if(!PyArg_ParseTuple(args, "iiiiii", &structType, &x1, &z1, &x2, &z2, &version)) return NULL;


    return Py_BuildValue("l",findStructure(structType, x1, z1, x2, z2, version));

}
PyDoc_STRVAR(findStructure_doc, "You have found hidden function");

static PyObject* pyStructureLower48(PyObject* self, PyObject* args)
{
	int structType, x1, z1, x2, z2, version;
	int64_t lower48;
    if(!PyArg_ParseTuple(args, "iliiiii", &structType, &lower48, &x1, &z1, &x2, &z2, &version)) return NULL;

	
	Pos q=structureLower48(structType, lower48, x1, z1, x2, z2, version); 
	
	if(q.x==30000000) return Py_BuildValue("");

	
    return Py_BuildValue("(ii)", q.x, q.z);
}
PyDoc_STRVAR(structureLower48_doc, "structureLower48(structType, lower48, x1, z1, x2, z2, version)->Tuple\n\n\
Returns Structure Position as a tuple if the lower 48 bits of the seed has the structure in the given region. Otherwise returns None");

static PyObject* pyIsValidStructurePos(PyObject* self, PyObject* args){
	int structType, structx, structz, version;
	int64_t seed; 
	if(!PyArg_ParseTuple(args, "iliii", &structType, &seed, &structx, &structz, &version)) return NULL;
	
	return Py_BuildValue("O", PyBool_FromLong(cIsViableStructurePos(structType, seed, structx, structz, version)));
}
PyDoc_STRVAR(isValidStructurePos_doc, "isValidStructurePos(structType, seed, structx, structz, version)->Bool\n\
Returns True if the given structure type could spawn at that location. Otherwise returns False");
static PyObject* pyGetStrongholds(PyObject* self, PyObject* args){
	int64_t seed;
	int N, version;
	

	if(!PyArg_ParseTuple(args,"lii", &seed, &N, &version)) return NULL;

	Pos* strongs=cgetStrongholds(seed, N, version);
	PyObject* stronglist=PyList_New(N);
	for(int i=0; i<N; i++){
		PyObject* tup=Py_BuildValue("(ii)", strongs[i].x, strongs[i].z);


		PyList_SET_ITEM(stronglist, i,  tup);

    }
    free(strongs);
	return stronglist;
}
PyDoc_STRVAR(getStrongholds_doc, "getStronghold(seed, N, version)->List\n\
Returns list of tuples that contains the the list of positions of the first [n] strongholds.");
static PyObject* pySaveMap(PyObject* self, PyObject* args){
	int64_t seed;
	int areaX, areaZ, areaWidth, areaHeight, scale, version; 
	char* filename;
	if (!PyArg_ParseTuple(args, "liiiiiis", &seed, &areaX, &areaZ, &areaWidth, &areaHeight, &scale, &version, &filename)) return NULL;

	return Py_BuildValue("O", PyBool_FromLong((long)cSaveMap(seed, areaX, areaZ, areaWidth, areaHeight, scale, version, filename)));
}
//meth>methods 
static PyMethodDef methods[] = {
    { "biome_at_pos", (PyCFunction) pyBiomeAtPos, METH_VARARGS, biomeAtPos_doc
	},
	{"biomes_in_area", pyBiomesInArea, METH_VARARGS, biomesInArea_doc},
	{"get_spawn", pyGetSpawn, METH_VARARGS, getSpawn_doc},
	{"find_structure", pyFindStructure, METH_VARARGS, findStructure_doc},
	{"structure_in_area", pyStructureLower48, METH_VARARGS,structureLower48_doc},
	{"is_valid_structure_pos", pyIsValidStructurePos, METH_VARARGS,isValidStructurePos_doc},
	{"get_strongholds", pyGetStrongholds, METH_VARARGS, getStrongholds_doc},
	{"save_map", pySaveMap, METH_VARARGS, PyDoc_STR("Saves a biome map of [seed] of an area. The area starts at (AreaX, AreaZ), and is [areaWidth] wide, and [areaHeight] wide.")},
    { NULL, NULL, 0, NULL }
};



// Module Definition Shit
static struct PyModuleDef overworld = {
    PyModuleDef_HEAD_INIT,
    "overworld",
    "Search functions for Pyubiomes, a Python wrapper for cubiomes.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_overworld(void)
{
    return PyModule_Create(&overworld);
}


