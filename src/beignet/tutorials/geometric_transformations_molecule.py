"""
Tutorial: 3D Geometric Transformations with Caffeine Molecule

This tutorial demonstrates the fundamental concepts of 3D geometric transformations
using a simplified caffeine molecule as a visually engaging example.

Topics covered:
- Quaternions - compact and efficient rotation representation
- Rotation matrices - standard matrix representation of rotations
- Euler angles - angle-based rotations
- Rotation vectors - axis-angle representation

All demonstrated using a beautiful caffeine molecule structure!
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

import beignet


def create_caffeine_molecule() -> tuple[
    torch.Tensor, list[tuple[int, int]], dict[str, torch.Tensor]
]:
    """
    Create a simplified caffeine molecule structure.

    Caffeine (C8H10N4O2) has a purine ring system with methyl groups.
    This is a simplified 3D representation for educational purposes.

    Returns
    -------
    atoms : torch.Tensor
        Atomic coordinates (N_atoms x 3)
    bonds : List[Tuple[int, int]]
        Bond connections as pairs of atom indices
    atom_types : Dict[str, torch.Tensor]
        Indices of different atom types
    """
    # Simplified caffeine structure coordinates (approximate)
    # Based on the purine ring system with methyl groups

    atom_coords = torch.tensor(
        [
            # Purine ring system (fused rings)
            [0.0, 0.0, 0.0],  # 0: N1
            [1.3, 0.0, 0.0],  # 1: C2
            [2.0, 1.2, 0.0],  # 2: N3
            [1.3, 2.4, 0.0],  # 3: C4
            [0.0, 2.4, 0.0],  # 4: C5
            [-0.7, 1.2, 0.0],  # 5: C6
            [0.0, 3.6, 0.0],  # 6: N7
            [1.3, 3.6, 0.0],  # 7: C8
            [2.0, 2.4, 0.0],  # 8: N9
            # Oxygen atoms (carbonyl groups)
            [2.0, -0.8, 0.0],  # 9: O1 (on C2)
            [-1.9, 1.2, 0.0],  # 10: O2 (on C6)
            # Methyl groups
            [-1.3, -0.8, 0.0],  # 11: C (methyl on N1)
            [-1.3, 4.4, 0.0],  # 12: C (methyl on N7)
            [2.0, 4.4, 0.0],  # 13: C (methyl on C8)
            # Selected hydrogen atoms for visualization
            [-1.3, -1.6, 0.5],  # 14: H (on methyl)
            [-1.3, -1.6, -0.5],  # 15: H (on methyl)
            [-2.1, -0.8, 0.0],  # 16: H (on methyl)
            [-1.3, 5.2, 0.5],  # 17: H (on methyl)
            [-1.3, 5.2, -0.5],  # 18: H (on methyl)
            [2.0, 5.2, 0.5],  # 19: H (on methyl)
            [2.0, 5.2, -0.5],  # 20: H (on methyl)
            # Ring hydrogen atoms
            [2.0, 4.4, 0.0],  # 21: H (on C8)
        ],
        dtype=torch.float32,
    )

    # Bond connections (atom index pairs)
    bonds = [
        # Purine ring bonds
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),  # 6-membered ring
        (4, 6),
        (6, 7),
        (7, 8),
        (8, 3),  # 5-membered ring
        # Carbonyl bonds
        (1, 9),  # C2=O
        (5, 10),  # C6=O
        # Methyl group bonds
        (0, 11),  # N1-CH3
        (6, 12),  # N7-CH3
        (7, 13),  # C8-CH3
        # Hydrogen bonds
        (11, 14),
        (11, 15),
        (11, 16),  # methyl H's
        (12, 17),
        (12, 18),  # methyl H's
        (13, 19),
        (13, 20),  # methyl H's
        (7, 21),  # ring H
    ]

    # Atom type indices
    atom_types = {
        "carbon": torch.tensor([1, 3, 4, 7, 11, 12, 13]),
        "nitrogen": torch.tensor([0, 2, 6, 8]),
        "oxygen": torch.tensor([9, 10]),
        "hydrogen": torch.tensor([14, 15, 16, 17, 18, 19, 20, 21]),
    }

    return atom_coords, bonds, atom_types


def plot_caffeine_molecule(
    atoms: torch.Tensor,
    bonds: list[tuple[int, int]],
    atom_types: dict[str, torch.Tensor],
    title: str = "Caffeine Molecule",
    ax: plt.Axes = None,
    colors: dict[str, str] = None,
    alpha: float = 0.8,
) -> plt.Axes:
    """
    Plot a caffeine molecule structure.

    Parameters
    ----------
    atoms : torch.Tensor
        Atomic coordinates
    bonds : List[Tuple[int, int]]
        Bond connections
    atom_types : Dict[str, torch.Tensor]
        Atom type indices
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib 3D axis to plot on
    colors : Dict[str, str], optional
        Colors for different atom types
    alpha : float
        Transparency level

    Returns
    -------
    ax : plt.Axes
        The matplotlib 3D axis
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    if colors is None:
        colors = {
            "carbon": "gray",
            "nitrogen": "blue",
            "oxygen": "red",
            "hydrogen": "white",
        }

    # Plot bonds first (so they appear behind atoms)
    for i, j in bonds:
        ax.plot(
            [atoms[i, 0], atoms[j, 0]],
            [atoms[i, 1], atoms[j, 1]],
            [atoms[i, 2], atoms[j, 2]],
            "k-",
            linewidth=2,
            alpha=alpha * 0.6,
        )

    # Plot atoms by type
    sizes = {"carbon": 150, "nitrogen": 120, "oxygen": 140, "hydrogen": 60}

    for atom_type, indices in atom_types.items():
        if len(indices) > 0:
            coords = atoms[indices]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=colors[atom_type],
                s=sizes[atom_type],
                alpha=alpha,
                label=f"{atom_type.capitalize()}",
                edgecolors="black",
                linewidth=1,
            )

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title)
    ax.legend()

    return ax


def compare_molecule_transformations(
    atoms: torch.Tensor,
    bonds: list[tuple[int, int]],
    atom_types: dict[str, torch.Tensor],
    rotation_type: str,
    rotation_params: torch.Tensor,
    title_prefix: str = "Caffeine Transformation",
) -> None:
    """
    Compare original and transformed caffeine molecule structures side by side.

    Parameters
    ----------
    atoms : torch.Tensor
        Original atomic coordinates
    bonds : List[Tuple[int, int]]
        Bond connections
    atom_types : Dict[str, torch.Tensor]
        Atom type indices
    rotation_type : str
        Type of rotation ('quaternion', 'matrix', 'euler', 'rotation_vector')
    rotation_params : torch.Tensor
        Rotation parameters
    title_prefix : str
        Prefix for the plot title
    """
    # Apply transformation based on type
    if rotation_type == "quaternion":
        atoms_rot = beignet.apply_quaternion(atoms, rotation_params)
    elif rotation_type == "matrix":
        atoms_rot = beignet.apply_rotation_matrix(atoms, rotation_params)
    elif rotation_type == "euler":
        atoms_rot = beignet.apply_euler_angle(atoms, rotation_params, axes="XYZ")
    elif rotation_type == "rotation_vector":
        atoms_rot = beignet.apply_rotation_vector(atoms, rotation_params)

    # Create side-by-side comparison
    fig = plt.figure(figsize=(16, 8))

    # Original molecule
    ax1 = fig.add_subplot(121, projection="3d")
    plot_caffeine_molecule(
        atoms, bonds, atom_types, title=f"{title_prefix} - Original", ax=ax1
    )

    # Transformed molecule
    ax2 = fig.add_subplot(122, projection="3d")
    plot_caffeine_molecule(
        atoms_rot, bonds, atom_types, title=f"{title_prefix} - Transformed", ax=ax2
    )

    # Set equal aspect ratio and limits
    for ax in [ax1, ax2]:
        ax.set_xlim([-3, 4])
        ax.set_ylim([-2, 6])
        ax.set_zlim([-2, 2])

    plt.tight_layout()
    plt.show()


def demonstrate_quaternions():
    """Demonstrate quaternion operations with caffeine molecule."""
    print("=" * 60)
    print("QUATERNION ROTATIONS WITH CAFFEINE MOLECULE")
    print("=" * 60)

    # Create caffeine molecule
    atoms, bonds, atom_types = create_caffeine_molecule()

    print("Created caffeine molecule (C8H10N4O2)")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Number of bonds: {len(bonds)}")
    print(f"Atom types: {list(atom_types.keys())}")

    # Create identity quaternion
    identity_quat = beignet.quaternion_identity(1)
    print(f"\nIdentity quaternion: {identity_quat}")

    # Create 45-degree rotation around Y-axis (good for viewing molecule)
    angle = np.pi / 4  # 45 degrees
    y_rotation_quat = torch.tensor(
        [[0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)]], dtype=torch.float32
    )

    print(f"45° Y-rotation quaternion: {y_rotation_quat}")
    print(f"Quaternion magnitude: {beignet.quaternion_magnitude(y_rotation_quat)}")

    # Visualize the rotation
    compare_molecule_transformations(
        atoms,
        bonds,
        atom_types,
        "quaternion",
        y_rotation_quat,
        "Quaternion Rotation (45° around Y-axis)",
    )

    # Demonstrate quaternion composition
    print("\n" + "-" * 40)
    print("QUATERNION COMPOSITION")
    print("-" * 40)

    # Create another rotation around X-axis
    x_rotation_quat = torch.tensor(
        [
            [np.sin(np.pi / 6), 0.0, 0.0, np.cos(np.pi / 6)]  # 30° around X
        ],
        dtype=torch.float32,
    )

    # Compose rotations
    composed_quat = beignet.compose_quaternion(y_rotation_quat, x_rotation_quat)
    print(f"Y-rotation: {y_rotation_quat}")
    print(f"X-rotation: {x_rotation_quat}")
    print(f"Composed: {composed_quat}")

    # Visualize composed rotation
    compare_molecule_transformations(
        atoms,
        bonds,
        atom_types,
        "quaternion",
        composed_quat,
        "Composed Quaternion Rotation (Y + X)",
    )


def demonstrate_rotation_matrices():
    """Demonstrate rotation matrix operations with caffeine molecule."""
    print("\n" + "=" * 60)
    print("ROTATION MATRICES WITH CAFFEINE MOLECULE")
    print("=" * 60)

    # Create caffeine molecule
    atoms, bonds, atom_types = create_caffeine_molecule()

    # Create rotation matrix for 60° around Z-axis
    angle = np.pi / 3  # 60 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    z_rotation_matrix = torch.tensor(
        [[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )

    print("60° Z-rotation matrix:")
    print(z_rotation_matrix)

    # Test matrix properties
    determinant = torch.det(z_rotation_matrix)
    is_orthogonal = torch.allclose(
        z_rotation_matrix @ z_rotation_matrix.transpose(-1, -2), torch.eye(3), atol=1e-6
    )

    print(f"Determinant: {determinant}")
    print(f"Is orthogonal: {is_orthogonal}")

    # Visualize the rotation
    compare_molecule_transformations(
        atoms,
        bonds,
        atom_types,
        "matrix",
        z_rotation_matrix,
        "Rotation Matrix (60° around Z-axis)",
    )

    # Convert quaternion to matrix and compare
    print("\n" + "-" * 40)
    print("QUATERNION TO MATRIX CONVERSION")
    print("-" * 40)

    quat = torch.tensor(
        [[0.0, 0.0, np.sin(np.pi / 6), np.cos(np.pi / 6)]], dtype=torch.float32
    )
    matrix_from_quat = beignet.quaternion_to_rotation_matrix(quat)

    print(f"Quaternion: {quat}")
    print("Converted to matrix:")
    print(matrix_from_quat)

    # Verify they produce the same result
    quat_result = beignet.apply_quaternion(atoms[:5], quat)  # Test with first 5 atoms
    matrix_result = beignet.apply_rotation_matrix(atoms[:5], matrix_from_quat)

    print(
        f"Results are identical: {torch.allclose(quat_result, matrix_result, atol=1e-6)}"
    )


def demonstrate_euler_angles():
    """Demonstrate Euler angle operations with caffeine molecule."""
    print("\n" + "=" * 60)
    print("EULER ANGLES WITH CAFFEINE MOLECULE")
    print("=" * 60)

    # Create caffeine molecule
    atoms, bonds, atom_types = create_caffeine_molecule()

    # Create Euler angles for interesting molecule viewing angle
    euler_angles = torch.tensor([[60.0, 30.0, 45.0]], dtype=torch.float32)  # degrees

    print(f"Euler angles (XYZ convention): {euler_angles} degrees")
    print(f"In radians: {torch.deg2rad(euler_angles)}")

    # Apply Euler rotation
    compare_molecule_transformations(
        atoms,
        bonds,
        atom_types,
        "euler",
        euler_angles,
        "Euler Angle Rotation (60°, 30°, 45°)",
    )

    # Demonstrate different conventions
    print("\n" + "-" * 40)
    print("DIFFERENT EULER CONVENTIONS")
    print("-" * 40)

    test_angles = torch.tensor([[45.0, 60.0, 30.0]], dtype=torch.float32)
    conventions = ["XYZ", "ZYX", "YXZ"]

    fig = plt.figure(figsize=(18, 6))

    for i, convention in enumerate(conventions):
        # Apply rotation with different convention
        if convention == "XYZ":
            rotated_atoms = beignet.apply_euler_angle(
                atoms, test_angles, axes="XYZ", degrees=True
            )
        elif convention == "ZYX":
            rotated_atoms = beignet.apply_euler_angle(
                atoms, test_angles, axes="ZYX", degrees=True
            )
        elif convention == "YXZ":
            rotated_atoms = beignet.apply_euler_angle(
                atoms, test_angles, axes="YXZ", degrees=True
            )

        # Plot
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        plot_caffeine_molecule(
            rotated_atoms, bonds, atom_types, title=f"Convention: {convention}", ax=ax
        )
        ax.set_xlim([-3, 4])
        ax.set_ylim([-2, 6])
        ax.set_zlim([-2, 2])

    plt.tight_layout()
    plt.show()


def demonstrate_rotation_vectors():
    """Demonstrate rotation vector operations with caffeine molecule."""
    print("\n" + "=" * 60)
    print("ROTATION VECTORS WITH CAFFEINE MOLECULE")
    print("=" * 60)

    # Create caffeine molecule
    atoms, bonds, atom_types = create_caffeine_molecule()

    # Create rotation vector for 90° around axis [1, 0, 1] (diagonal)
    angle = np.pi / 2  # 90 degrees
    axis = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    axis = axis / torch.norm(axis)  # normalize

    rotation_vector = angle * axis.unsqueeze(0)

    print(f"Rotation vector: {rotation_vector}")
    print(
        f"Magnitude (angle): {torch.norm(rotation_vector)} rad = {torch.rad2deg(torch.norm(rotation_vector)):.1f}°"
    )
    print(f"Axis: {axis}")

    # Visualize the rotation
    compare_molecule_transformations(
        atoms,
        bonds,
        atom_types,
        "rotation_vector",
        rotation_vector,
        "Rotation Vector (90° around [1,0,1])",
    )

    # Convert to other representations
    print("\n" + "-" * 40)
    print("CONVERSION TO OTHER REPRESENTATIONS")
    print("-" * 40)

    quat_from_rotvec = beignet.rotation_vector_to_quaternion(rotation_vector)
    matrix_from_rotvec = beignet.rotation_vector_to_rotation_matrix(rotation_vector)
    euler_from_rotvec = beignet.rotation_vector_to_euler_angle(
        rotation_vector, axes="XYZ"
    )

    print(f"To quaternion: {quat_from_rotvec}")
    print(f"To Euler angles: {torch.rad2deg(euler_from_rotvec)} degrees")

    # Verify all give same result
    rotvec_result = beignet.apply_rotation_vector(atoms[:5], rotation_vector)
    quat_result = beignet.apply_quaternion(atoms[:5], quat_from_rotvec)
    matrix_result = beignet.apply_rotation_matrix(atoms[:5], matrix_from_rotvec)

    print(
        f"RotVec vs Quaternion: {torch.allclose(rotvec_result, quat_result, atol=1e-5)}"
    )
    print(
        f"RotVec vs Matrix: {torch.allclose(rotvec_result, matrix_result, atol=1e-5)}"
    )


def demonstrate_chemical_application():
    """Demonstrate a realistic chemical application: molecular docking orientation."""
    print("\n" + "=" * 60)
    print("CHEMICAL APPLICATION: MOLECULAR DOCKING")
    print("=" * 60)

    # Create caffeine molecule
    atoms, bonds, atom_types = create_caffeine_molecule()

    print("Simulating molecular docking orientations...")
    print("Different orientations of caffeine approaching a binding site")

    # Define several docking orientations
    orientations = [
        (
            "Approach 1",
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        ),  # original
        ("Approach 2", torch.tensor([[90.0, 0.0, 0.0]], dtype=torch.float32)),  # 90° X
        ("Approach 3", torch.tensor([[0.0, 90.0, 0.0]], dtype=torch.float32)),  # 90° Y
        (
            "Approach 4",
            torch.tensor([[45.0, 45.0, 0.0]], dtype=torch.float32),
        ),  # tilted
        (
            "Approach 5",
            torch.tensor([[0.0, 0.0, 180.0]], dtype=torch.float32),
        ),  # flipped
        (
            "Approach 6",
            torch.tensor([[30.0, 60.0, 90.0]], dtype=torch.float32),
        ),  # complex
    ]

    # Create subplots for different orientations
    fig = plt.figure(figsize=(18, 12))

    for i, (name, euler_angles) in enumerate(orientations):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")

        # Apply rotation
        if torch.allclose(euler_angles, torch.tensor([[0.0, 0.0, 0.0]])):
            rotated_atoms = atoms
        else:
            rotated_atoms = beignet.apply_euler_angle(
                atoms, euler_angles, axes="XYZ", degrees=True
            )

        # Plot
        plot_caffeine_molecule(rotated_atoms, bonds, atom_types, title=name, ax=ax)
        ax.set_xlim([-3, 4])
        ax.set_ylim([-2, 6])
        ax.set_zlim([-2, 2])

    plt.tight_layout()
    plt.show()

    # Calculate molecular descriptors for each orientation
    print("\nMolecular orientation analysis:")
    for name, euler_angles in orientations[:3]:  # Show first 3
        if torch.allclose(euler_angles, torch.tensor([[0.0, 0.0, 0.0]])):
            rotated_atoms = atoms
        else:
            rotated_atoms = beignet.apply_euler_angle(
                atoms, euler_angles, axes="XYZ", degrees=True
            )

        # Calculate center of mass
        center_of_mass = torch.mean(rotated_atoms, dim=0)

        # Calculate moment of inertia tensor (simplified)
        centered_atoms = rotated_atoms - center_of_mass
        inertia = torch.sum(centered_atoms**2, dim=0)

        print(f"{name}: COM = {center_of_mass}, Inertia = {inertia}")


def run_complete_tutorial():
    """Run the complete caffeine molecular transformations tutorial."""
    print("☕ CAFFEINE MOLECULE GEOMETRIC TRANSFORMATIONS TUTORIAL ☕")
    print("=" * 70)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create initial molecule for overview
    atoms, bonds, atom_types = create_caffeine_molecule()

    print("Welcome to the Caffeine Molecular Geometric Transformations Tutorial!")
    print(
        "This tutorial demonstrates 3D rotations using a caffeine molecule structure."
    )
    print(f"Caffeine (C8H10N4O2) with {len(atoms)} atoms and {len(bonds)} bonds")

    # Show initial molecule structure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_caffeine_molecule(
        atoms, bonds, atom_types, title="Caffeine Molecule - Starting Structure", ax=ax
    )
    ax.set_xlim([-3, 4])
    ax.set_ylim([-2, 6])
    ax.set_zlim([-2, 2])
    plt.show()

    # Run all demonstrations
    demonstrate_quaternions()
    demonstrate_rotation_matrices()
    demonstrate_euler_angles()
    demonstrate_rotation_vectors()
    demonstrate_chemical_application()

    print("\n" + "=" * 70)
    print("🎉 TUTORIAL COMPLETE! 🎉")
    print("=" * 70)
    print("\nKey takeaways:")
    print("✓ Caffeine molecule provides a great 3D structure for visualizing rotations")
    print("✓ Quaternions are perfect for smooth molecular rotations")
    print("✓ Rotation matrices provide clear mathematical understanding")
    print("✓ Euler angles are intuitive for molecular orientations")
    print("✓ Rotation vectors offer compact representation")
    print("✓ All representations are interconvertible")
    print("✓ 3D transformations are essential for molecular docking and drug design")
    print(
        "\nNext steps: Try creating your own molecules or modify the caffeine structure!"
    )


if __name__ == "__main__":
    run_complete_tutorial()
