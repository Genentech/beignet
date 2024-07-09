import mdtraj
from beignet.datasets import TrajectoryDataset
from mdtraj import Trajectory


class TestTrajectoryDataset:
    def test___init__(self, data_path):
        dataset = TrajectoryDataset(
            mdtraj.load_pdb,
            "pdb",
            data_path,
        )

        assert dataset.root is not None

    def test___getitem__(self, data_path):
        dataset = TrajectoryDataset(
            mdtraj.load_pdb,
            "pdb",
            data_path,
        )

        assert isinstance(dataset[0], Trajectory)

    def test___len__(self, data_path):
        dataset = TrajectoryDataset(
            mdtraj.load_pdb,
            "pdb",
            data_path,
        )

        assert len(dataset) == 1
