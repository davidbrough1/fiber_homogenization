import scipy as sp
import numpy as np
import pickle
from matplotlib.cbook import flatten


def printNodeSet(f, nodeSet, nodeSetName):
    nl = "\n"
    f.writelines(('*****************************************************', nl))
    f.writelines(('*NSET, NSET={}'.format(nodeSetName), nl))
    i = 0
    for x in sorted(nodeSet):
        f.write('{},   '.format(x))
        i += 1
        if (i % 16 == 0):
            f.write('\n')
    f.write('\n')


def printEleSet(f, eleSet, eleSetName):
    nl = "\n"
    f.writelines(('*****************************************************', nl))
    f.writelines(('*ELSET, ELSET={}'.format(eleSetName), nl))
    i = 0
    for x in sorted(eleSet):
        f.write('{},   '.format(x))
        i += 1
        if (i % 16 == 0):
            f.write('\n')
    f.write('\n')


# Returns the node number for the given indices
def getEleNumber(i, j, k, intx, inty, intz, Fine_int):
    return (j) + (i)*inty*Fine_int*intz*Fine_int + (k)*inty*Fine_int + 1


def getIndices(eleNumber, dim_length):
    i = eleNumber - 1
    layer_size = dim_length**2
    y = i % dim_length
    z = (i % layer_size)/dim_length
    x = i/layer_size
    return (x, y, z)


# Returning the sets of nodes on the two faces and the edge
def intFaceNodes(intx, inty, intz, setA, setC, setE,
                 Row_ELM_f, Layer_ELM_f, Block_ELM_f):
    NodeintYp = setC
    NodeYp = []
    # For n2plus
    for i in range((intx + 1)*(intz + 1)):
        NodeYp.append(NodeintYp + i*Layer_ELM_f[1])
    # For n2neg
    NodeintYn = setA
    NodeYn = []
    for i in range((intx + 1)*(intz + 1)):
        NodeYn.append(NodeintYn + i*Layer_ELM_f[1])
    # For n1plus
    NodeintXp = setE
    NodeXp = []
    for i in range((inty + 1)*(intz + 1)):
        NodeXp.append(NodeintXp + i*Row_ELM_f[1])
    # For n1neg
    NodeintXn = setA
    NodeXn = []
    for i in range((inty + 1)*(intz + 1)):
        NodeXn.append(NodeintXn + i*Row_ELM_f[1])
    # For n3plus
    step = 1
    NodeZp = []
    NodeZp2 = []
    for i in range(inty + 1):
        Node1 = step + (Layer_ELM_f[1] * intz)
        NodeZp.append(Node1)
        for i in range(intx + 1):
            NodeZp2.append(Node1 + (Row_ELM_f[1]*i))
        step = step + Block_ELM_f[1]
    NodeZp.extend(NodeZp2)

    # For n3neg
    step = setA
    NodeZn = []
    NodeZn2 = []
    for i in range(inty + 1):
        Node1 = step
        NodeZn.append(Node1)
        for i in range(intx + 1):
            NodeZn2.append(Node1 + (Row_ELM_f[1]*i))
        step = step + Block_ELM_f[1]
    NodeZn.extend(NodeZn2)

    # For -x Normal
    NodeXn_pbc = set(NodeXn).difference(set(NodeZn), set(NodeZp),
                                        set(NodeYp), set(NodeYn))
    # For +x Normal
    NodeXp_pbc = set(NodeXp).difference(set(NodeZn), set(NodeZp),
                                        set(NodeYp), set(NodeYn))
    # For +y Normal
    NodeYp_pbc = set(NodeYp).difference(set(NodeZn), set(NodeZp),
                                        set(NodeXn), set(NodeXp))
    # For -y Normal
    NodeYn_pbc = set(NodeYn).difference(set(NodeZn), set(NodeZp),
                                        set(NodeXn), set(NodeXp))
    # For -z Normal
    NodeZn_pbc = set(NodeZn).difference(set(NodeYn), set(NodeYp),
                                        set(NodeXn), set(NodeXp))
    # For +z Normal
    NodeZp_pbc = set(NodeZp).difference(set(NodeYn), set(NodeYp),
                                        set(NodeXn), set(NodeXp))

    # For Edges
    n3minus_n1minus = set(NodeZn).intersection(
        set(NodeXn)).difference(set(NodeYp), set(NodeYn))
    n3minus_n1plus = set(NodeZn).intersection(
        set(NodeXp)).difference(set(NodeYp), set(NodeYn))
    n2minus_n3minus = set(NodeYn).intersection(
        set(NodeZn)).difference(set(NodeXp), set(NodeXn))
    n2plus_n3minus = set(NodeYp).intersection(
        set(NodeZn)).difference(set(NodeXp), set(NodeXn))
    n3plus_n1minus = set(NodeZp).intersection(
        set(NodeXn)).difference(set(NodeYp), set(NodeYn))
    n3plus_n1plus = set(NodeZp).intersection(
        set(NodeXp)).difference(set(NodeYp), set(NodeYn))
    n2minus_n3plus = set(NodeYn).intersection(
        set(NodeZp)).difference(set(NodeXp), set(NodeXn))
    n2plus_n3plus = set(NodeYp).intersection(
        set(NodeZp)).difference(set(NodeXp), set(NodeXn))
    n1plus_n2minus = set(NodeXp).intersection(
        set(NodeYn)).difference(set(NodeZp), set(NodeZn))
    n1plus_n2plus = set(NodeXp).intersection(
        set(NodeYp)).difference(set(NodeZp), set(NodeZn))
    n1minus_n2minus = set(NodeXn).intersection(
        set(NodeYn)).difference(set(NodeZp), set(NodeZn))
    n1minus_n2plus = set(NodeXn).intersection(
        set(NodeYp)).difference(set(NodeZp), set(NodeZn))

    return (NodeXn_pbc, NodeXp_pbc, NodeYp_pbc, NodeYn_pbc,
            NodeZp_pbc, NodeZn_pbc, n3minus_n1minus, n3minus_n1plus,
            n2minus_n3minus, n2plus_n3minus, n3plus_n1minus, n3plus_n1plus,
            n2minus_n3plus, n2plus_n3plus, n1plus_n2minus, n1plus_n2plus,
            n1minus_n2minus, n1minus_n2plus)


# Generates an abaqus input for the given microstructure.
# Only works for microstructures with two phases
def generateAbaqusInp(inputFileName, ms, elastic_modulus=(120, 80),
                      poissions_ratio=(0.3, 0.3), viscoelastic=False,
                      Prony_values=[(0.0769, 0.0769, 0.75), (0.2, 0.2, 1.5)],
                      n_steps=1000000):
    """Genrateds an abaqus input file

    Args:
        inputFileName: name of input file
        ms: microstructure
        elastic_modulus: elastic moduli for different phases
        poissions_ratio: poissions ratios for phases
        viscoelastic: indicate if viscoelastic simulation should be used
        Prony_values: Prony_values used for viscoelastic simulation
        n_steps: number of time steps used in simulation

    """

    f = open(inputFileName + ".microstructure", 'w')
    pickle.dump(ms, f)
    f.close()
    f = open(inputFileName, 'w')
    nl = "\n"
    headerLines = '*Preprint, echo=NO, model=No, history=NO, contact=NO', nl, '*Heading', nl
    headerLines += '****************************************************', nl
    headerLines += '*node', nl

    f.writelines(headerLines)

    shape = ms.shape
    Length_X = shape[0] - 1
    Length_Y = shape[1] - 1
    Length_Z = shape[2] - 1

    intx = shape[0]
    inty = shape[1]
    intz = shape[2]

    Fine_int = 1

    NodeNo1 = 1
    ElementNo1 = 1
    x0 = 0
    y0 = 0
    z0 = 0

    setA = NodeNo1
    setC = setA + (inty*Fine_int)
    setB = setA + (inty*Fine_int + 1) * (intz*Fine_int)
    setD = setA + (inty*Fine_int + 1) * (intz*Fine_int) + (inty*Fine_int)

    LayerNodes = (inty*Fine_int + 1) * (intz*Fine_int) + (inty*Fine_int) + 1

    setE = (intx*Fine_int)*LayerNodes + setA
    setG = setE + (inty*Fine_int)
    setF = setE + (inty*Fine_int + 1)*(intz*Fine_int)
    setH = setE + (inty*Fine_int + 1)*(intz*Fine_int) + (inty*Fine_int)

    Total_Nodes = setH

    modelCoordinates = [[setA, x0, y0, z0], [setB, x0, y0, z0 + Length_Z],
                        [setC, x0, y0 + Length_Y, z0],
                        [setD, x0, y0 + Length_Y, z0 + Length_Z],
                        [setE, x0 + Length_X, y0, z0],
                        [setF, x0 + Length_X, y0, z0 + Length_Z],
                        [setG, x0 + Length_X, y0 + Length_Y, z0],
                        [setH, x0 + Length_X, y0 + Length_Y, z0 + Length_Z]]
    vertexLineFormat = '{}, {}, {}, {} \n'
    for i in range(len(modelCoordinates)):
        f.writelines(vertexLineFormat.format(*modelCoordinates[i]))

    nSetNames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    nSetFormat = '*NSET, NSET={} \n {} \n'
    f.writelines(('*****************************************************', nl))
    for i in range(len(nSetNames)):
        f.writelines(nSetFormat.format(nSetNames[i], modelCoordinates[i][0]))

    # Generate nodes for main region
    # Below: [x, y] = x for interval & y for numbering
    nFillDict = {}
    setAB_f = [intz, (inty*Fine_int + 1)*Fine_int]
    nFillDict['AB'] = setAB_f
    setCD_f = [intz, (inty*Fine_int + 1)*Fine_int]
    nFillDict['CD'] = setCD_f
    setABCD_f = [inty, Fine_int]
    nFillDict['ABCD'] = setABCD_f

    nFillFormat = '*Nfill, Nset={}\n{}, {}, {}, {}\n'
    f.writelines(('****************************************************', nl))

    setEF_f = [intz, (inty*Fine_int + 1)*Fine_int]
    nFillDict['EF'] = setEF_f
    setGH_f = [intz, (inty*Fine_int + 1)*Fine_int]
    nFillDict['GH'] = setGH_f
    setEFGH_f = [inty, Fine_int]
    nFillDict['EFGH'] = setEFGH_f

    setABCDEFGH_f = [intx, LayerNodes*Fine_int]
    nFillDict['ABCDEFGH'] = setABCDEFGH_f

    f.write(nFillFormat.format('AB', 'A', 'B', setAB_f[0], setAB_f[1]))
    f.write(nFillFormat.format('CD', 'C', 'D', setCD_f[0], setCD_f[1]))
    f.write(nFillFormat.format('ABCD', 'AB', 'CD', setABCD_f[0], setABCD_f[1]))
    f.write(nFillFormat.format('EF', 'E', 'F', setEF_f[0], setEF_f[1]))
    f.write(nFillFormat.format('GH', 'G', 'H', setGH_f[0], setGH_f[1]))
    f.write(nFillFormat.format('EFGH', 'EF', 'GH', setEFGH_f[0], setEFGH_f[1]))
    f.write(nFillFormat.format('ABCDEFGH', 'ABCD', 'EFGH', setABCDEFGH_f[0],
                               setABCDEFGH_f[1]))

    f.writelines(('****************************************************', nl))
    f.write(nFillFormat.format('BD', 'B', 'D', setABCD_f[0], setABCD_f[1]))
    f.write(nFillFormat.format('FH', 'F', 'H', setABCD_f[0], setABCD_f[1]))
    f.write(nFillFormat.format('BDFH', 'BD', 'FH', setABCDEFGH_f[0],
                               setABCDEFGH_f[1]))

    f.writelines(('*****************************************************', nl))
    f.writelines(('** Bottom Face Node Set', nl))
    f.write(nFillFormat.format('AC', 'A', 'C', setABCD_f[0], setABCD_f[1]))
    f.write(nFillFormat.format('EG', 'E', 'G', setABCD_f[0], setABCD_f[1]))
    f.write(nFillFormat.format('ACEG', 'AC', 'EG', setABCDEFGH_f[0],
                               setABCDEFGH_f[1]))

    # Define Element#1
    Cube_1 = setA
    Cube_2 = setA+Fine_int
    Cube_3 = setA+(inty*Fine_int+2)*Fine_int
    Cube_4 = setA+(inty*Fine_int+1)*Fine_int
    Cube_5 = (1*Fine_int)*LayerNodes+setA
    Cube_6 = (1*Fine_int)*LayerNodes+setA+Fine_int
    Cube_7 = Cube_5+(inty*Fine_int+2)*Fine_int
    Cube_8 = Cube_5+(inty*Fine_int+1)*Fine_int

    Element1 = [ElementNo1, Cube_1, Cube_2, Cube_3, Cube_4,
                Cube_5, Cube_6, Cube_7, Cube_8]

    f.writelines(('*****************************************************', nl))
    f.writelines(('*ELEMENT, TYPE=C3D8', nl))
    f.writelines((','.join(map(str, Element1)), nl))

    # Generate Elements for main region
    # Below: [x y z] = x no of elements including master
    #                  y increment in node number
    #                  z increment in element number

    Row_ELM_f = [inty, Fine_int, Fine_int]
    Layer_ELM_f = [intz, setAB_f[1], (inty*Fine_int)*Fine_int]
    Block_ELM_f = [intx, setABCDEFGH_f[1],
                   (inty*Fine_int)*(intz*Fine_int)*Fine_int]

    ELGEN = [ElementNo1, Row_ELM_f, Layer_ELM_f, Block_ELM_f]

    f.writelines(('*****************************************************', nl))
    f.writelines(('*ELGEN, elset=allel', nl))
    f.writelines((','.join(map(str, flatten(ELGEN))), nl))

    elset1 = []
    elset2 = []
    for i in range(intx):
        for j in range(inty):
            for k in range(intz):
                phase = ms[i][j][k]
                if (phase == 1):
                    elset1.append(getEleNumber(i, j, k, intx,
                                               inty, intz, Fine_int))
                elif (phase == 2):
                    elset2.append(getEleNumber(i, j, k, intx,
                                               inty, intz, Fine_int))
                else:
                    print 'phase not 1 or 2: {}'.format(phase)
                    elset2.append(getEleNumber(i, j, k, intx, inty,
                                               intz, Fine_int))

    printEleSet(f, elset1, 'elset1')
    printEleSet(f, elset2, 'elset2')

    Total_Elements=intx*inty*intz

    (NodeXn_pbc, NodeXp_pbc, NodeYp_pbc, NodeYn_pbc, NodeZp_pbc, NodeZn_pbc,
     n3minus_n1minus, n3minus_n1plus, n2minus_n3minus, n2plus_n3minus,
     n3plus_n1minus, n3plus_n1plus, n2minus_n3plus, n2plus_n3plus,
     n1plus_n2minus, n1plus_n2plus, n1minus_n2minus, n1minus_n2plus) = \
        intFaceNodes(intx, inty, intz, setA, setC, setE, Row_ELM_f,
                     Layer_ELM_f, Block_ELM_f)

    printNodeSet(f, NodeXp_pbc, 'n1plus')
    printNodeSet(f, NodeXn_pbc, 'n1minus')
    printNodeSet(f, NodeYn_pbc, 'n2minus')
    printNodeSet(f, NodeYp_pbc, 'n2plus')
    printNodeSet(f, NodeYn_pbc, 'n2minus')
    printNodeSet(f, NodeZn_pbc, 'n3minus')
    printNodeSet(f, NodeZp_pbc, 'n3plus')

    printNodeSet(f, n3minus_n1minus, 'n3minus_n1minus')
    printNodeSet(f, n3minus_n1plus, 'n3minus_n1plus')
    printNodeSet(f, n2minus_n3minus, 'n2minus_n3minus')
    printNodeSet(f, n2plus_n3minus, 'n2plus_n3minus')
    printNodeSet(f, n3plus_n1minus, 'n3plus_n1minus')
    printNodeSet(f, n3plus_n1plus, 'n3plus_n1plus')
    printNodeSet(f, n2minus_n3plus, 'n2minus_n3plus')
    printNodeSet(f, n2plus_n3plus, 'n2plus_n3plus')
    printNodeSet(f, n1plus_n2minus, 'n1plus_n2minus')
    printNodeSet(f, n1plus_n2plus, 'n1plus_n2plus')
    printNodeSet(f, n1minus_n2minus, 'n1minus_n2minus')
    printNodeSet(f, n1minus_n2plus, 'n1minus_n2plus')

    # Implement Boundary conditions
    # Equation
    f.writelines(('*****************************************************', nl))
    f.writelines(('** Implement Periodic Boundary Conditions', nl))
    f.writelines(('*Equation', nl))

    f.writelines(('3', nl))
    f.writelines(('n1plus, 1, 1, n1minus, 1, -1, {}, 1, -1'.format(setH), nl))
    f.writelines(('2', nl))
    f.writelines(('n1plus, 2, 1, n1minus, 2, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n1plus, 3, 1, n1minus, 3, -1', nl))
    f.writelines(('**', nl))
    f.writelines(('2', nl))
    f.writelines(('n2plus, 1, 1, n2minus, 1, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n2plus, 2, 1, n2minus, 2, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n2plus, 3, 1, n2minus, 3, -1', nl))
    f.writelines(('**', nl))
    f.writelines(('2', nl))
    f.writelines(('n3plus, 1, 1, n3minus, 1, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n3plus, 2, 1, n3minus, 2, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n3plus, 3, 1, n3minus, 3, -1', nl))

    f.writelines(('**', nl))
    f.writelines(('3', nl))
    f.writelines(('n1plus_n2plus, 1, 1, n1minus_n2plus, 1, -1, {}, 1, -1'.format(setH), nl))
    f.writelines(('3', nl))
    f.writelines(('n1plus_n2minus, 1, 1, n1minus_n2minus, 1, -1, {}, 1, -1'.format(setH), nl))
    f.writelines(('2', nl))
    f.writelines(('n1minus_n2plus, 1, 1, n1minus_n2minus, 1, -1', nl))

    f.writelines(('**', nl))
    f.writelines(('2', nl))
    f.writelines(('n1plus_n2plus, 2, 1, n1minus_n2plus, 2, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n1plus_n2minus, 2, 1, n1minus_n2minus, 2, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n1minus_n2plus, 2, 1, n1minus_n2minus, 2, -1', nl))
    f.writelines(('**', nl))
    f.writelines(('2', nl))
    f.writelines(('n1plus_n2plus, 3, 1, n1minus_n2plus, 3, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n1plus_n2minus, 3, 1, n1minus_n2minus, 3, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n1minus_n2plus, 3, 1, n1minus_n2minus, 3, -1', nl))
    f.writelines(('**', nl))
    f.writelines(('**', nl))
    f.writelines(('3', nl))
    # maybe do formatting to fill in value for H
    f.writelines(('n3plus_n1plus, 1, 1, n3plus_n1minus, 1, -1, {}, 1, -1'.format(setH), nl))
    f.writelines(('3', nl))
    f.writelines(('n3minus_n1plus, 1, 1, n3minus_n1minus, 1, -1, {}, 1, -1'.format(setH), nl))
    f.writelines(('2', nl))
    f.writelines(('n3plus_n1minus, 1, 1, n3minus_n1minus, 1, -1', nl))
    f.writelines(('**', nl))
    f.writelines(('2', nl))
    f.writelines(('n3plus_n1plus, 2, 1, n3plus_n1minus, 2, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n3minus_n1plus, 2, 1, n3minus_n1minus, 2, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n3plus_n1minus, 2, 1, n3minus_n1minus, 2, -1', nl))
    f.writelines(('**', nl))
    f.writelines(('2', nl))
    f.writelines(('n3plus_n1plus, 3, 1, n3plus_n1minus, 3, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n3minus_n1plus, 3, 1, n3minus_n1minus, 3, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n3plus_n1minus, 3, 1, n3minus_n1minus, 3, -1', nl))
    f.writelines(('**', nl))
    f.writelines(('**', nl))
    f.writelines(('2', nl))
    f.writelines(('n2plus_n3plus, 1, 1, n2minus_n3plus, 1, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n2plus_n3minus, 1, 1, n2minus_n3minus, 1, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n2minus_n3plus, 1, 1, n2minus_n3minus, 1, -1', nl))
    f.writelines(('**', nl))
    f.writelines(('2', nl))
    f.writelines(('n2plus_n3plus, 2, 1, n2minus_n3plus, 2, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n2plus_n3minus, 2, 1, n2minus_n3minus, 2, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n2minus_n3plus, 2, 1, n2minus_n3minus, 2, -1', nl))
    f.writelines(('**', nl))
    f.writelines(('2', nl))
    f.writelines(('n2plus_n3plus, 3, 1, n2minus_n3plus, 3, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n2plus_n3minus, 3, 1, n2minus_n3minus, 3, -1', nl))
    f.writelines(('2', nl))
    f.writelines(('n2minus_n3plus, 3, 1, n2minus_n3minus, 3, -1', nl))

    f.writelines(('**** ----------------------------------------------------------------- ', nl))
    f.writelines(('** MATERIALS', nl))
    f.writelines(('**', nl))
    f.writelines(('*Solid Section, elset=elset1, material=material-1', nl))
    if (not viscoelastic):
        f.writelines(('1.,', nl))
    f.writelines(('*Material, name=material-1', nl))
    if (viscoelastic):
        f.writelines(('*Elastic, moduli=Long Term', nl))
        f.writelines((str(elastic_modulus[0]) + ',' +
                      str(poissions_ratio[0]), nl))
        f.writelines(('*Viscoelastic, Time=Prony', nl))
        f.writelines((str(Prony_values[0][1:-1]), nl))
    else:
        f.writelines(('*Elastic,type=isotropic', nl))
        f.writelines((str(elastic_modulus[0]) + ',' +
                      str(poissions_ratio[0]), nl))
    f.writelines(('** Solid (element 2 = elset2)', nl))
    f.writelines(('**', nl))
    f.writelines(('*Solid Section, elset=elset2, material=material-2', nl))
    if (not viscoelastic):
        f.writelines(('1.,', nl))
    f.writelines(('**', nl))
    f.writelines(('*Material, name=material-2', nl))
    if (viscoelastic):
        f.writelines(('*Elastic, moduli=Long Term', nl))
        f.writelines((str(elastic_modulus[1]) + ',' +
                      str(poissions_ratio[1]), nl))
        f.writelines(('*Viscoelastic, Time=Prony', nl))
        f.writelines((str(Prony_values[1][1:-1]), nl))
    else:
        f.writelines(('*Elastic,type=isotropic', nl))
        f.writelines((str(elastic_modulus[1]) + ',' +
                      str(poissions_ratio[1]), nl))

    f.writelines(('** ----------------------------------------------------------------', nl))
    f.writelines(('**     ', nl))

    f.writelines(('** ----------------------------------------------------------------', nl))
    f.writelines(('** ', nl))
    if (viscoelastic):
        # do something other than what was happening here
        # f.writelines(('*Amplitude, Name=disp1, Definition=Tabular', nl))
        # f.writelines(('10,1', nl))
        f.writelines(('*Step, Amplitude=Step, name=Step-1,  nlgeom=YES, inc=1000000', nl))
        f.writelines(('*Visco', nl))
        f.writelines(('0.5,10,1e-05,0.5', nl))
        f.writelines(('** Name: BC-1 Type: Displacement/Rotation', nl))
        f.writelines(('*Boundary', nl))
        f.writelines(('**', nl))
        f.writelines(('{},1,3,0'.format(setA), nl))
        f.writelines(('{},1,3,0'.format(setB), nl))
        f.writelines(('{},1,3,0'.format(setC), nl))
        f.writelines(('{},1,3,0'.format(setD), nl))
        f.writelines(('{},1,1,0.02'.format(setE), nl))
        f.writelines(('{},1,1,0.02'.format(setF), nl))
        f.writelines(('{},1,1,0.02'.format(setG), nl))
        f.writelines(('{},1,1,0.02'.format(setH), nl))
        f.writelines(('{},2,3,0'.format(setE), nl))
        f.writelines(('{},2,3,0'.format(setF), nl))
        f.writelines(('{},2,3,0'.format(setG), nl))
        f.writelines(('{},2,3,0'.format(setH), nl))
        f.writelines(('** ', nl))
        f.writelines(('** OUTPUT REQUESTS', nl))
        f.writelines(('**', nl))
        f.writelines(('*output, field, frequency=0', nl))
        f.writelines(('**', nl))
        f.writelines(('*output, history, frequency=0', nl))
        f.writelines(('** ', nl))
        f.writelines(('*el print, summary=no, totals=yes', nl))
        f.writelines(('E', nl))
        f.writelines(('*End Step', nl))

        # step 2
#        f.writelines(('*Amplitude, Name=disp2, Definition=Tabular', nl))
#        f.writelines(('10,1', nl))
        f.writelines(('*Step, Amplitude=Step, name=Step-2,  nlgeom=YES, inc=' +
                      str(n_steps), nl))
        f.writelines(('*Visco', nl))
        f.writelines(('0.5,10,1e-05,0.5', nl))
        f.writelines(('** Name: BC-1 Type: Displacement/Rotation', nl))
        f.writelines(('*Boundary', nl))
        f.writelines(('**', nl))
        f.writelines(('{},1,3,0'.format(setA), nl))
        f.writelines(('{},1,3,0'.format(setB), nl))
        f.writelines(('{},1,3,0'.format(setC), nl))
        f.writelines(('{},1,3,0'.format(setD), nl))
        f.writelines(('{},1,1,-0.02'.format(setE), nl))
        f.writelines(('{},1,1,-0.02'.format(setF), nl))
        f.writelines(('{},1,1,-0.02'.format(setG), nl))
        f.writelines(('{},1,1,-0.02'.format(setH), nl))
        f.writelines(('{},2,3,0'.format(setE), nl))
        f.writelines(('{},2,3,0'.format(setF), nl))
        f.writelines(('{},2,3,0'.format(setG), nl))
        f.writelines(('{},2,3,0'.format(setH), nl))
        f.writelines(('** ', nl))
        f.writelines(('** OUTPUT REQUESTS', nl))
        f.writelines(('**', nl))
        f.writelines(('*output, field, frequency=0', nl))
        f.writelines(('**', nl))
        f.writelines(('*output, history, frequency=0', nl))
        f.writelines(('** ', nl))
        f.writelines(('*el print, summary=no, totals=yes', nl))
        f.writelines(('E', nl))
        f.writelines(('*End Step', nl))
    else:
        f.writelines(('** STEP: Step-1', nl))
        f.writelines(('** ', nl))
        f.writelines(('*Step, name=Step-1', nl))
        f.writelines(('*Static', nl))
        f.writelines(('1., 1., 1e-05, 1.', nl))
        f.writelines(('** ', nl))
        f.writelines(('** BOUNDARY CONDITIONS', nl))
        f.writelines(('** ', nl))
        f.writelines(('** Name: BC-1 Type: Displacement/Rotation', nl))
        f.writelines(('*Boundary', nl))
        f.writelines(('**', nl))
        f.writelines(('{},1,3,0'.format(setA), nl))
        f.writelines(('{},1,3,0'.format(setB), nl))
        f.writelines(('{},1,3,0'.format(setC), nl))
        f.writelines(('{},1,3,0'.format(setD), nl))
        f.writelines(('{},1,1,0.02'.format(setE), nl))
        f.writelines(('{},1,1,0.02'.format(setF), nl))
        f.writelines(('{},1,1,0.02'.format(setG), nl))
        f.writelines(('{},1,1,0.02'.format(setH), nl))
        f.writelines(('{},2,3,0'.format(setE), nl))
        f.writelines(('{},2,3,0'.format(setF), nl))
        f.writelines(('{},2,3,0'.format(setG), nl))
        f.writelines(('{},2,3,0'.format(setH), nl))
        f.writelines(('** ', nl))
        f.writelines(('** OUTPUT REQUESTS', nl))
        f.writelines(('**', nl))
        f.writelines(('*output, field, frequency=0', nl))
        f.writelines(('**', nl))
        f.writelines(('*output, history, frequency=0', nl))
        f.writelines(('** ', nl))
        f.writelines(('*el print, summary=no, totals=yes', nl))
        f.writelines(('E', nl))
        f.writelines(('**', nl))
        f.writelines(('*End Step', nl))
    f.close()
