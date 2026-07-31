"""Case-preserving dependency materialization path migration."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

from ...utils.path_security import ensure_path_within, validate_path_segments

if TYPE_CHECKING:
    from ...install.resolution_staging import ResolutionStagingSession
    from .reference import DependencyReference


class MaterializationPathCollisionError(ValueError):
    """Raised when one canonical identity has multiple physical paths."""


class MaterializationPathReader(Protocol):
    """Filesystem read seam for platform-neutral casing tests."""

    def exists(self, path: Path) -> bool:
        """Return whether *path* exists."""

    def is_dir(self, path: Path) -> bool:
        """Return whether *path* is a directory."""

    def iterdir(self, path: Path) -> Iterable[Path]:
        """Yield children of *path*."""

    def samefile(self, left: Path, right: Path) -> bool:
        """Return whether two spellings address the same entry."""


class _NativeMaterializationPathReader:
    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_dir(self, path: Path) -> bool:
        return path.is_dir()

    def iterdir(self, path: Path) -> Iterable[Path]:
        return path.iterdir()

    def samefile(self, left: Path, right: Path) -> bool:
        return os.path.samefile(left, right)


_NATIVE_READER = _NativeMaterializationPathReader()


def build_materialization_path(
    dependency: DependencyReference,
    apm_modules_dir: Path,
) -> Path:
    """Build the source-cased path for one dependency below ``apm_modules``."""
    if dependency.is_marketplace:
        raise ValueError(
            "Cannot compute install path for unresolved marketplace dependency "
            f"'{dependency.marketplace_plugin_name}@{dependency.marketplace_name}'"
        )

    if dependency.is_local and dependency.local_path:
        pkg_dir_name = Path(dependency.local_path).name
        validate_path_segments(
            pkg_dir_name,
            context="local package path",
            reject_empty=True,
        )
        if dependency.declaring_parent:
            identity = dependency.anchored_local_path or dependency.local_path
            parent_slot = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            result = apm_modules_dir / "_local" / parent_slot / pkg_dir_name
        else:
            result = apm_modules_dir / "_local" / pkg_dir_name
        ensure_path_within(result, apm_modules_dir)
        return result

    repo_parts = dependency.repo_url.split("/")
    validate_path_segments(dependency.repo_url, context="repo_url")
    if dependency.virtual_path:
        validate_path_segments(dependency.virtual_path, context="virtual_path")

    result: Path | None = None
    if dependency.is_virtual:
        if dependency.is_virtual_subdirectory():
            if dependency.is_azure_devops() and len(repo_parts) >= 3:
                result = apm_modules_dir.joinpath(
                    repo_parts[0],
                    repo_parts[1],
                    repo_parts[2],
                    dependency.virtual_path,
                )
            elif len(repo_parts) >= 2:
                result = apm_modules_dir.joinpath(
                    *repo_parts,
                    dependency.virtual_path,
                )
        else:
            package_name = dependency.get_virtual_package_name()
            if dependency.is_azure_devops() and len(repo_parts) >= 3:
                result = apm_modules_dir / repo_parts[0] / repo_parts[1] / package_name
            elif len(repo_parts) >= 2:
                result = apm_modules_dir / repo_parts[0] / package_name
    elif dependency.is_azure_devops() and len(repo_parts) >= 3:
        result = apm_modules_dir / repo_parts[0] / repo_parts[1] / repo_parts[2]
    elif len(repo_parts) >= 2:
        result = apm_modules_dir.joinpath(*repo_parts)

    if result is None:
        result = apm_modules_dir.joinpath(*repo_parts)
    ensure_path_within(result, apm_modules_dir)
    return result


def _relative_parts(desired: Path, apm_modules_dir: Path) -> tuple[str, ...]:
    """Validate and return *desired* components below ``apm_modules``."""
    ensure_path_within(desired, apm_modules_dir)
    try:
        relative = desired.absolute().relative_to(apm_modules_dir.absolute())
    except ValueError as exc:
        raise MaterializationPathCollisionError(
            f"Materialization path escapes apm_modules: {desired}"
        ) from exc
    return relative.parts


def _candidate_matches(
    desired_parts: Sequence[str],
    candidate_relatives: Iterable[PurePosixPath],
) -> list[PurePosixPath]:
    """Return candidates with the same component-wise casefolded identity."""
    expected = tuple(part.casefold() for part in desired_parts)
    return sorted(
        (
            candidate
            for candidate in candidate_relatives
            if tuple(part.casefold() for part in candidate.parts) == expected
        ),
        key=lambda value: value.as_posix(),
    )


def find_case_equivalent_materialization_path(
    desired: Path,
    apm_modules_dir: Path,
    *,
    reader: MaterializationPathReader = _NATIVE_READER,
    candidate_relatives: Iterable[PurePosixPath] | None = None,
) -> Path | None:
    """Find the sole existing path whose components differ only by casing."""
    desired_parts = _relative_parts(desired, apm_modules_dir)
    if candidate_relatives is not None:
        matches = _candidate_matches(desired_parts, candidate_relatives)
        if len(matches) > 1:
            rendered = ", ".join(match.as_posix() for match in matches)
            raise MaterializationPathCollisionError(
                f"Found multiple materialization paths for one package identity: {rendered}"
            )
        return apm_modules_dir.joinpath(*matches[0].parts) if matches else None

    current = apm_modules_dir
    for desired_part in desired_parts:
        if not reader.is_dir(current):
            return None
        matches = sorted(
            (
                child
                for child in reader.iterdir(current)
                if child.name.casefold() == desired_part.casefold()
            ),
            key=lambda child: child.name,
        )
        if len(matches) > 1:
            rendered = ", ".join(str(match) for match in matches)
            raise MaterializationPathCollisionError(
                f"Found multiple materialization paths for one package identity: {rendered}"
            )
        if not matches:
            return None
        current = matches[0]
    return current


def _relocate_case_components(
    existing: Path,
    desired: Path,
    apm_modules_dir: Path,
    staging_session: ResolutionStagingSession,
    reader: MaterializationPathReader,
) -> None:
    """Rename case-different components top-down through the transaction."""
    existing_parts = _relative_parts(existing, apm_modules_dir)
    desired_parts = _relative_parts(desired, apm_modules_dir)
    current_parent = apm_modules_dir
    for existing_part, desired_part in zip(existing_parts, desired_parts, strict=True):
        source = current_parent / existing_part
        target = current_parent / desired_part
        if existing_part != desired_part:
            if reader.exists(target):
                if not reader.samefile(source, target):
                    raise MaterializationPathCollisionError(
                        f"Materialization path collision between {source} and {target}"
                    )
            staging_session.relocate_path(source, target)
        current_parent = target


def prepare_materialization_path(
    dependency: DependencyReference,
    apm_modules_dir: Path,
    staging_session: ResolutionStagingSession,
    *,
    reader: MaterializationPathReader = _NATIVE_READER,
) -> Path:
    """Return the display-cased path, transactionally migrating stale casing."""
    desired = dependency.get_install_path(apm_modules_dir)
    if not dependency.has_case_insensitive_repo_identity:
        return desired

    existing = find_case_equivalent_materialization_path(
        desired,
        apm_modules_dir,
        reader=reader,
    )
    if existing is None or existing == desired:
        return desired
    _relocate_case_components(
        existing,
        desired,
        apm_modules_dir,
        staging_session,
        reader,
    )
    return desired
