import pytest
from biotite.database import rcsb


@pytest.fixture
def structure_7k7r_pdb():
    return rcsb.fetch("7k7r", "pdb")


@pytest.fixture
def structure_7k7r_cif():
    return rcsb.fetch("7k7r", "cif")


@pytest.fixture
def structure_7k7r_bcif():
    return rcsb.fetch("7k7r", "bcif")


# has insertion codes
@pytest.fixture
def structure_1s78_pdb():
    return rcsb.fetch("1s78", "pdb")


@pytest.fixture
def gapped_aho_7k7r():
    return {
        "A": "DVVLTQSPLSLPVILGQPASISCRSS--QSLVYSD-GRTYLNWFQQRPGQSPRRLIYK--------ISKRDSGVPERFSGSGSG--TDFTLEISRVEAEDVGIYYCMQGSH-----------------------WPVTFGQGTKVEIKR",
        "B": "-VQLVES-GGGLVKPGGSLRLSCVSSG-FTFSN-----YWMSWVRQAPGGGLEWVANINQD---GSEKYYVDSVKGRFTSSRDNTKNSLFLQLNSLRAEDTGIYYCTRDPP-----------------------YFDNWGQGTLVTVSS",
        "D": "DVVLTQSPLSLPVILGQPASISCRSS--QSLVYSD-GRTYLNWFQQRPGQSPRRLIYK--------ISKRDSGVPERFSGSGSG--TDFTLEISRVEAEDVGIYYCMQGSH-----------------------WPVTFGQGTKVEIKR",
        "E": "QVQLVES-GGGLVKPGGSLRLSCVSSG-FTFSN-----YWMSWVRQAPGGGLEWVANINQD---GSEKYYVDSVKGRFTSSRDNTKNSLFLQLNSLRAEDTGIYYCTRDPP-----------------------YFDNWGQGTLVTVSS",
    }
