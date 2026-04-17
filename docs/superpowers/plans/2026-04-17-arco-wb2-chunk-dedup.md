# ARCO/WB2 Zarr Chunk Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate duplicate Zarr chunk downloads when multiple pressure levels of the same variable are fetched at the same time in ARCO and WB2 data sources.

**Architecture:** The current `fetch()` loop creates a cartesian product of `(time, variable)` pairs and calls `zarr_array.getitem((time_idx, level_idx))` independently for each pair. Since ARCO/WB2 chunks span all levels at a single timestep (`chunk-1` = 1 time step, all levels), fetching `u500`, `u850`, `u200` at the same time downloads the same chunk 3×. The fix groups requests by `(zarr_variable_name, time_index)` and issues one batched `getitem` per group.

**Tech Stack:** Python 3.11+, zarr v3 async API, asyncio, numpy, pytest, pytest-asyncio (already in dev deps)

---

## Root Cause Recap

```
ARCO zarr store layout:  u_component_of_wind  →  [time=700k, level=37, lat=721, lon=1440]
                                                   chunk shape:  [1,       37,     721,    1440]
```

`fetch("u500")` → `getitem((time_idx, 21))` → downloads the whole `[1, 37, 721, 1440]` chunk  
`fetch("u850")` → `getitem((time_idx, 30))` → downloads the **same chunk** again  
`fetch("u200")` → `getitem((time_idx, 14))` → downloads the **same chunk** again  

Since all 3 are fired concurrently via `asyncio.gather`, the chunk cache can't deduplicate them.

---

## File Map

| File | Change |
|---|---|
| `earth2studio/data/arco.py` | Rewrite `ARCO.fetch()`; add `from collections import defaultdict` |
| `earth2studio/data/wb2.py` | Rewrite `_WB2Base.fetch()` and `WB2Climatology.fetch()`; add `from collections import defaultdict` |
| `test/data/test_arco.py` | Add `test_arco_no_duplicate_chunk_fetch` unit test |
| `test/data/test_wb2era5.py` | Add `test_wb2_no_duplicate_chunk_fetch` unit test |

Note: `fetch_wrapper` and `fetch_array` are **kept** in all classes — they are an established pattern across 6+ data sources and `WB2Climatology.fetch()` inherits `fetch_wrapper` from `_WB2Base`.

---

## Task 1: Write failing unit test for ARCO

**Files:**
- Modify: `test/data/test_arco.py`

- [ ] **Step 1: Add imports at top of test file**

In `test/data/test_arco.py`, add after the existing imports:

```python
from unittest.mock import AsyncMock, MagicMock
```

- [ ] **Step 2: Add the failing test at the bottom of the file**

```python
@pytest.mark.asyncio
async def test_arco_no_duplicate_chunk_fetch():
    """Multiple levels of same variable at same time must result in exactly one
    zarr getitem call for the atmospheric array, not one per level."""
    from earth2studio.data import ARCO

    # Construct ARCO bypassing _async_init (no network needed)
    ds = ARCO.__new__(ARCO)
    ds._cache = False
    ds._verbose = False
    ds.async_timeout = 60
    ds.level_coords = np.array(
        [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150,
         175, 200, 225, 250, 300, 350, 400, 450, 500, 550,
         600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
         925, 950, 975, 1000],
        dtype=np.int32,
    )
    ds.ml_level_coords = np.array([], dtype=np.int32)

    atmo_getitem_calls = []

    async def mock_atmo_getitem(key):
        atmo_getitem_calls.append(key)
        # Batch fetch: (time_idx, slice(None)) → [37, 721, 1440]
        if isinstance(key, tuple) and isinstance(key[1], slice):
            return np.zeros((37, 721, 1440), dtype=np.float32)
        # Single level: (time_idx, level_idx) → [721, 1440]
        return np.zeros((721, 1440), dtype=np.float32)

    mock_atmo = MagicMock()
    mock_atmo.shape = (700000, 37, 721, 1440)
    mock_atmo.getitem = mock_atmo_getitem

    async def mock_group_get(name):
        return mock_atmo

    mock_group = MagicMock()
    mock_group.get = mock_group_get

    mock_ml_group = MagicMock()
    mock_ml_group.get = AsyncMock(return_value=mock_atmo)

    ds.zarr_group = mock_group
    ds.ml_zarr_group = mock_ml_group

    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)

    result = await ds.fetch(
        [datetime.datetime(2020, 1, 1, 0, 0)],
        ["u500", "u850", "u200"],
    )

    # Three levels of the same variable at the same time → exactly 1 zarr fetch
    assert len(atmo_getitem_calls) == 1, (
        f"Expected 1 chunk fetch for 3 levels of the same variable, got {len(atmo_getitem_calls)}"
    )
    assert result.shape == (1, 3, 721, 1440)
    assert not np.isnan(result.values).any()
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /Users/solario/Projects/Solido/earth2studio
pytest test/data/test_arco.py::test_arco_no_duplicate_chunk_fetch -v
```

Expected: `FAILED` — current code calls `getitem` 3 times, `atmo_getitem_calls` will have 3 entries.

---

## Task 2: Fix `ARCO.fetch()` to deduplicate chunk downloads

**Files:**
- Modify: `earth2studio/data/arco.py`

- [ ] **Step 1: Add `defaultdict` to stdlib imports**

In `earth2studio/data/arco.py`, change the imports block to add `from collections import defaultdict`:

```python
import asyncio
import functools
import os
import pathlib
import re
import shutil
from collections import defaultdict
from datetime import datetime
```

- [ ] **Step 2: Replace `ARCO.fetch()` method (lines 215–271)**

Replace only the `fetch` method body (keep `fetch_wrapper` and `fetch_array` as-is below it):

```python
    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the ARCO lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from ARCO
        """
        if self.zarr_group is None:
            raise ValueError(
                "Zarr group is not initialized! If you are calling this \
            function directly make sure the data source is initialized inside the async \
            loop!"
            )

        time, variable = prep_data_inputs(time, variable)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)
        self._validate_time(time)

        xr_array = xr.DataArray(
            data=np.empty(
                (len(time), len(variable), len(self.ARCO_LAT), len(self.ARCO_LON))
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": self.ARCO_LAT,
                "lon": self.ARCO_LON,
            },
        )

        # Group requests by (zarr_variable_name, is_model_level, time_index, time_xr_idx)
        # so that multiple levels of the same variable at the same time share one fetch.
        # Each value is a list of (level_index_or_None, var_xr_idx, modifier).
        groups = defaultdict(list)
        for j, v in enumerate(variable):
            try:
                arco_name, modifier = ARCOLexicon[v]
            except KeyError as e:
                logger.error(f"variable id {v} not found in ARCO lexicon")
                raise e
            parts = arco_name.split("::")
            zarr_var = parts[0]
            level = parts[1] if len(parts) > 1 else None
            is_ml = self._is_mdl_level(v)
            lcoords = self.ml_level_coords if is_ml else self.level_coords
            level_idx = int(np.searchsorted(lcoords, int(level))) if level else None
            for i, t in enumerate(time):
                time_idx = self._get_time_index(t)
                groups[(zarr_var, is_ml, time_idx, i)].append((level_idx, j, modifier))

        async def fetch_group(key: tuple, entries: list) -> None:
            zarr_var, is_ml, time_idx, xr_i = key
            zgroup = self.ml_zarr_group if is_ml else self.zarr_group
            zarr_array = await zgroup.get(zarr_var)
            shape = zarr_array.shape
            logger.debug(
                f"Fetching ARCO zarr array '{zarr_var}' at time_idx={time_idx} "
                f"({len(entries)} variable(s))"
            )
            if len(shape) == 2:  # static variable
                data = await zarr_array.getitem(slice(None))
                for _, j, modifier in entries:
                    xr_array[xr_i, j] = modifier(data)
            elif len(shape) == 3:  # surface variable
                data = await zarr_array.getitem(time_idx)
                for _, j, modifier in entries:
                    xr_array[xr_i, j] = modifier(data)
            else:  # atmospheric variable
                if len(entries) == 1:
                    level_idx, j, modifier = entries[0]
                    data = await zarr_array.getitem((time_idx, level_idx))
                    xr_array[xr_i, j] = modifier(data)
                else:
                    # Multiple levels of the same variable at the same timestep:
                    # fetch all levels in one call (one chunk) and slice locally.
                    data = await zarr_array.getitem((time_idx, slice(None)))
                    for level_idx, j, modifier in entries:
                        xr_array[xr_i, j] = modifier(data[level_idx])

        await tqdm.gather(
            *[fetch_group(key, entries) for key, entries in groups.items()],
            desc="Fetching ARCO data",
            disable=(not self._verbose),
        )
        return xr_array
```

- [ ] **Step 3: Run the unit test to verify it now passes**

```bash
pytest test/data/test_arco.py::test_arco_no_duplicate_chunk_fetch -v
```

Expected: `PASSED`

- [ ] **Step 4: Commit**

```bash
git add earth2studio/data/arco.py test/data/test_arco.py
git commit -m "fix(data): deduplicate Zarr chunk fetches in ARCO for multi-level requests

When multiple pressure levels of the same variable are requested at the same
time, the old code issued one zarr getitem per level, re-downloading the same
chunk (which spans all levels) once per request. The new fetch() groups
requests by (zarr_var, time_index) and issues a single getitem for each group,
slicing levels from the result locally.

Fixes #778"
```

---

## Task 3: Write failing unit test for WB2

**Files:**
- Modify: `test/data/test_wb2era5.py`

- [ ] **Step 1: Add imports**

In `test/data/test_wb2era5.py`, add after existing imports:

```python
from unittest.mock import AsyncMock, MagicMock
```

- [ ] **Step 2: Add the failing test at the bottom of the file**

```python
@pytest.mark.asyncio
async def test_wb2_no_duplicate_chunk_fetch():
    """Multiple levels of same variable at same time must result in exactly one
    zarr getitem call for the atmospheric array."""
    from earth2studio.data import WB2ERA5

    # WB2ERA5.__new__(WB2ERA5) inherits the class-level WB2_ERA5_LAT/LON from WB2ERA5
    ds = WB2ERA5.__new__(WB2ERA5)
    ds._zarr_store_name = "test.zarr"
    ds._product = "era5"
    ds._cache = False
    ds._verbose = False
    ds.async_timeout = 60
    ds.level_coords = np.array(
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
        dtype=np.int32,
    )

    atmo_getitem_calls = []

    async def mock_atmo_getitem(key):
        atmo_getitem_calls.append(key)
        if isinstance(key, tuple) and isinstance(key[1], slice):
            return np.zeros((13, 721, 1440), dtype=np.float32)
        return np.zeros((721, 1440), dtype=np.float32)

    mock_atmo = MagicMock()
    mock_atmo.shape = (23412, 13, 721, 1440)
    mock_atmo.getitem = mock_atmo_getitem

    async def mock_group_get(name):
        return mock_atmo

    mock_group = MagicMock()
    mock_group.get = mock_group_get
    ds.zarr_group = mock_group

    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)

    result = await ds.fetch(
        [datetime.datetime(2020, 1, 1, 0, 0)],
        ["u500", "u850", "u200"],
    )

    assert len(atmo_getitem_calls) == 1, (
        f"Expected 1 chunk fetch for 3 levels of the same variable, got {len(atmo_getitem_calls)}"
    )
    assert result.shape == (1, 3, 721, 1440)
    assert not np.isnan(result.values).any()
```

- [ ] **Step 3: Check what imports test_wb2era5.py already has**

```bash
head -20 test/data/test_wb2era5.py
```

Make sure `pathlib` and `datetime` are already imported (they likely are). If not, add them.

- [ ] **Step 4: Run test to verify it fails**

```bash
pytest test/data/test_wb2era5.py::test_wb2_no_duplicate_chunk_fetch -v
```

Expected: `FAILED` — current code calls `getitem` 3 times.

---

## Task 4: Fix `_WB2Base.fetch()` to deduplicate chunk downloads

**Files:**
- Modify: `earth2studio/data/wb2.py`

- [ ] **Step 1: Add `defaultdict` to stdlib imports**

In `earth2studio/data/wb2.py`, add `from collections import defaultdict` to the imports block:

```python
import asyncio
import functools
import inspect
import os
import pathlib
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Literal
```

- [ ] **Step 2: Replace only `_WB2Base.fetch()` (lines 150–211), keeping `fetch_wrapper` and `fetch_array` intact**

`fetch_wrapper` must be kept — `WB2Climatology.fetch()` (at line 619) inherits and calls `self.fetch_wrapper` until Task 5 replaces it.

Replace just the `fetch` method:

```python
    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the WB2 lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from weather bench 2
        """
        if self.zarr_group is None:
            raise ValueError(
                "Zarr group is not initialized! If you are calling this \
            function directly make sure the data source is initialized inside the async \
            loop!"
            )

        time, variable = prep_data_inputs(time, variable)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)
        self._validate_time(time)

        xr_array = xr.DataArray(
            data=np.empty(
                (
                    len(time),
                    len(variable),
                    len(self.WB2_ERA5_LAT),
                    len(self.WB2_ERA5_LON),
                )
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": self.WB2_ERA5_LAT,
                "lon": self.WB2_ERA5_LON,
            },
        )

        # Group requests by (zarr_variable_name, time_index, time_xr_idx) so that
        # multiple levels of the same variable at the same time share one fetch.
        # Each value is a list of (level_index_or_None, var_xr_idx, modifier).
        groups = defaultdict(list)
        for j, v in enumerate(variable):
            try:
                wb2_name, modifier = WB2Lexicon[v]  # type: ignore
            except KeyError as e:
                logger.error(f"variable id {v} not found in WB2 lexicon")
                raise e
            zarr_var, level = wb2_name.split("::")
            level_idx = (
                int(np.searchsorted(self.level_coords, int(level))) if level else None
            )
            for i, t in enumerate(time):
                time_idx = self._get_time_index(t)
                groups[(zarr_var, time_idx, i)].append((level_idx, j, modifier))

        async def fetch_group(key: tuple, entries: list) -> None:
            zarr_var, time_idx, xr_i = key
            zarr_array = await self.zarr_group.get(zarr_var)
            shape = zarr_array.shape
            logger.debug(
                f"Fetching WB2 zarr array '{zarr_var}' at time_idx={time_idx} "
                f"({len(entries)} variable(s))"
            )
            if len(shape) == 2:  # static
                data = await zarr_array.getitem(slice(None))
                for _, j, modifier in entries:
                    out = modifier(data)
                    if out.shape[0] > out.shape[1]:
                        out = np.flip(out, axis=-1).T
                    xr_array[xr_i, j] = out
            elif len(shape) == 3:  # surface
                data = await zarr_array.getitem(time_idx)
                for _, j, modifier in entries:
                    out = modifier(data)
                    if out.shape[0] > out.shape[1]:
                        out = np.flip(out, axis=-1).T
                    xr_array[xr_i, j] = out
            else:  # atmospheric
                if len(entries) == 1:
                    level_idx, j, modifier = entries[0]
                    data = await zarr_array.getitem((time_idx, level_idx))
                    out = modifier(data)
                    if out.shape[0] > out.shape[1]:
                        out = np.flip(out, axis=-1).T
                    xr_array[xr_i, j] = out
                else:
                    # Fetch all levels once; slice locally.
                    data = await zarr_array.getitem((time_idx, slice(None)))
                    for level_idx, j, modifier in entries:
                        out = modifier(data[level_idx])
                        if out.shape[0] > out.shape[1]:
                            out = np.flip(out, axis=-1).T
                        xr_array[xr_i, j] = out

        await tqdm.gather(
            *[fetch_group(key, entries) for key, entries in groups.items()],
            desc="Fetching WB2 data",
            disable=(not self._verbose),
        )
        return xr_array
```

**Why the shape flip is needed in every branch:** Some lower-resolution WB2 zarr stores (e.g., 32×64) save arrays as `[lon, lat]` instead of `[lat, lon]`. The original `fetch_array` applies `np.flip(output, axis=-1).T` when `output.shape[0] > output.shape[1]`. This must be preserved for all variable types in the new grouped code.

- [ ] **Step 3: Run WB2 unit test to verify it passes**

```bash
pytest test/data/test_wb2era5.py::test_wb2_no_duplicate_chunk_fetch -v
```

Expected: `PASSED`

---

## Task 5: Fix `WB2Climatology.fetch()` for the same issue

**Files:**
- Modify: `earth2studio/data/wb2.py` (WB2Climatology class only)

- [ ] **Step 1: Verify the current WB2Climatology.fetch() structure**

```bash
grep -n "fetch_wrapper\|fetch_group\|fetch_array\|async def fetch" earth2studio/data/wb2.py
```

Confirm `WB2Climatology.fetch()` currently calls `self.fetch_wrapper` (inherited from `_WB2Base`). After this task, it will use `fetch_group` instead.

- [ ] **Step 2: Replace `WB2Climatology.fetch()` with grouped version**

The climatology index key is `(zarr_var, hour_idx, day_idx, xr_i)` because it uses `(hour_index, day_of_year_index)` instead of a single `time_index`.

Replace the entire `WB2Climatology.fetch()` method:

```python
    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the WB2 Climatology lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from weather bench 2
        """
        if self.zarr_group is None:
            raise ValueError(
                "Zarr group is not initialized! If you are calling this \
            function directly make sure the data source is initialized inside the async \
            loop!"
            )

        time, variable = prep_data_inputs(time, variable)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)
        self._validate_time(time)

        if inspect.isawaitable(self.zarr_group):
            self.zarr_group = await self.zarr_group

        WB2_CLIMATE_LAT = await (await self.zarr_group.get("latitude")).getitem(
            slice(None)
        )
        WB2_CLIMATE_LON = await (await self.zarr_group.get("longitude")).getitem(
            slice(None)
        )
        self.level_coords = await (await self.zarr_group.get("level")).getitem(
            slice(None)
        )

        xr_array = xr.DataArray(
            data=np.empty(
                (len(time), len(variable), len(WB2_CLIMATE_LAT), len(WB2_CLIMATE_LON))
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": WB2_CLIMATE_LAT[:],
                "lon": WB2_CLIMATE_LON[:],
            },
        )

        # Group by (zarr_var, hour_idx, day_idx, time_xr_idx) to deduplicate chunk fetches.
        # Each value is a list of (level_index_or_None, var_xr_idx, modifier).
        groups = defaultdict(list)
        for j, v in enumerate(variable):
            try:
                wb2_name, modifier = WB2ClimatetologyLexicon[v]  # type: ignore
            except KeyError as e:
                logger.error(f"variable id {v} not found in WB2 lexicon")
                raise e
            zarr_var, level = wb2_name.split("::")
            level_idx = (
                int(np.searchsorted(self.level_coords, int(level))) if level else None
            )
            for i, t in enumerate(time):
                hour_idx, day_idx = self._get_time_index(t)
                groups[(zarr_var, hour_idx, day_idx, i)].append((level_idx, j, modifier))

        async def fetch_group(key: tuple, entries: list) -> None:
            zarr_var, hour_idx, day_idx, xr_i = key
            zarr_array = await self.zarr_group.get(zarr_var)
            shape = zarr_array.shape
            logger.debug(
                f"Fetching WB2 climatology '{zarr_var}' at hour={hour_idx} day={day_idx} "
                f"({len(entries)} variable(s))"
            )
            # Surface variable: [hour_idx(4), day_of_year(366), lat, lon]
            if len(shape) == 4:
                data = await zarr_array.getitem((hour_idx, day_idx))
                for _, j, modifier in entries:
                    xr_array[xr_i, j] = modifier(data)
            # Atmospheric variable: [hour_idx(4), day_of_year(366), level, lat, lon]
            else:
                if len(entries) == 1:
                    level_idx, j, modifier = entries[0]
                    data = await zarr_array.getitem((hour_idx, day_idx, level_idx))
                    xr_array[xr_i, j] = modifier(data)
                else:
                    # Fetch all levels once; slice locally.
                    data = await zarr_array.getitem((hour_idx, day_idx, slice(None)))
                    for level_idx, j, modifier in entries:
                        xr_array[xr_i, j] = modifier(data[level_idx])

        await tqdm.gather(
            *[fetch_group(key, entries) for key, entries in groups.items()],
            desc="Fetching WB2 climatology data",
            disable=(not self._verbose),
        )
        return xr_array
```

- [ ] **Step 3: Verify test collection succeeds (no import errors)**

```bash
pytest test/data/test_wb2climate.py -v --collect-only
```

Expected: tests collected without errors.

- [ ] **Step 4: Commit**

```bash
git add earth2studio/data/wb2.py test/data/test_wb2era5.py
git commit -m "fix(data): deduplicate Zarr chunk fetches in WB2 for multi-level requests

Same fix as ARCO: group atmospheric requests by (zarr_var, time_index) and
issue one getitem per group, slicing levels locally from the result.
Applies to _WB2Base.fetch() and WB2Climatology.fetch().

Fixes #778"
```

---

## Task 6: Lint check

**Files:** None modified — verification only.

- [ ] **Step 1: Run all linters**

```bash
cd /Users/solario/Projects/Solido/earth2studio
make lint
```

Or use: `/lint`

Expected: ruff, mypy, and pre-commit checks all pass.

- [ ] **Step 2: Run license header check**

Use: `/license`

Expected: all Python files in `earth2studio/` have the SPDX Apache-2.0 header (no new files were created, so this should be clean).

---

## Task 7: Final verification

- [ ] **Step 1: Run all non-slow tests in the affected test files**

```bash
pytest test/data/test_arco.py test/data/test_wb2era5.py test/data/test_wb2climate.py -v -k "not slow"
```

Expected: All tests pass, including `test_arco_no_duplicate_chunk_fetch` and `test_wb2_no_duplicate_chunk_fetch`.

- [ ] **Step 2: Commit any lint fixes if needed**

```bash
git add -u
git commit -m "chore: fix lint issues from chunk dedup refactor"
```

---

## Cross-Validation Notes

These items were verified before the plan was written:

| Claim | Verified |
|---|---|
| `ARCOLexicon[v]` returns `(str, Callable)` | ✓ `LexiconType.__getitem__` → `get_item` → `(arco_key, mod)` |
| `asyncio_mode = "strict"` in pyproject.toml | ✓ line 415; `@pytest.mark.asyncio` per-test is correct |
| `pytest-asyncio` already in dev deps | ✓ pyproject.toml line 353 |
| `fetch_wrapper` used by WB2Climatology.fetch() | ✓ wb2.py:619 calls `self.fetch_wrapper` |
| `fetch_wrapper` is a codebase-wide pattern | ✓ present in goes, hrrr, gefs, gfs, ncar, wb2 |
| `from loguru import logger` in wb2.py | ✓ present |
| WB2 shape-flip logic applies to all variable types | ✓ original `fetch_array` applies it after all branches |
| `WB2ERA5.__new__(WB2ERA5)` inherits `WB2_ERA5_LAT` | ✓ class attribute on WB2ERA5 subclass |

### Known Edge Cases Handled

- **Single level**: uses existing `getitem((time_idx, level_idx))` path — no behavior change
- **Multiple levels same variable, different times**: different `time_idx` → different group keys → separate fetches ✓
- **Multiple different variables, same time**: different `zarr_var` → different group keys → separate fetches ✓
- **Static/surface variables**: no `level_idx`; grouped by `(zarr_var, time_idx)` → fetched once ✓
- **WB2 shape flip**: applied per-entry in every branch of `fetch_group` ✓
- **ARCO model-level variables**: routed via `is_ml` flag in group key ✓
- **WB2Climatology `fetch_wrapper` inheritance**: not removed; still callable for backward compat ✓
