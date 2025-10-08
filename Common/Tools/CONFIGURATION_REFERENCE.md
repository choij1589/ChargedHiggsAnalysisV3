# Configuration Reference for plotter.py

This document provides a comprehensive reference for all configuration parameters supported by the canvas classes in `plotter.py`.

## Overview

All canvas classes (`ComparisonCanvas`, `KinematicCanvas`, `KinematicCanvasWithRatio`) now inherit from `BaseCanvas`, which provides unified configuration handling using `config.get()` with sensible defaults.

## Configuration Parameters

### Common Parameters (All Canvas Classes)

These parameters are supported by all three canvas classes.

| Parameter | Type | Required/Default | Description | Notes |
|-----------|------|------------------|-------------|-------|
| `era` | str | **REQUIRED** | Data-taking era | Examples: "2022", "2022EE", "2023", "2023BPix", "Run2", "Run3" |
| `CoM` | float | *Default: auto* | Center-of-mass energy in TeV | Auto-detected from era: 13 TeV (Run2), 13.6 TeV (Run3) |
| `xTitle` | str | *Default: ""* | X-axis title | Supports ROOT LaTeX syntax (e.g., `"p_{T} [GeV]"`) |
| `yTitle` | str | *Default: "Events"* | Y-axis title | |
| `xRange` | list | **REQUIRED** | X-axis range or bin edges | `[xmin, xmax]` for simple range, or `[edge1, edge2, ..., edgeN]` for variable binning |
| `yRange` | list | *Default: auto* | Y-axis range | `[ymin, ymax]`. Auto-calculated from histogram content if not specified |
| `rebin` | int | *Default: none* | Fixed-width rebinning factor | Combines every N bins. Ignored if variable binning is used |
| `overflow` | bool | *Default: False* | Handle overflow bins | **NEW**: Now available in ALL classes. Accumulates bins beyond xRange into last visible bin (only for asymmetric ranges) |
| `logy` | bool | *Default: False* | Use logarithmic Y-axis | Automatically adjusts Y-range for log scale |
| `normalize` | bool | *Default: False* | Normalize histograms to unit area | Includes overflow/underflow bins |
| `prescaled` | bool | *Default: False* | Mark as prescaled trigger | Removes luminosity label, adds "Prescaled" to era text |
| `maxDigits` | int | *Default: 3* | Maximum digits on Y-axis | Controls ROOT axis label formatting |
| `legend` | list | *Default: auto* | Legend position | `[x1, y1, x2, y2, textSize, columns]`. Auto-positioned if not specified |
| `systSrc` | str | *Default: "Stat+Syst"* | Systematics legend label | **NEW**: Now available in ALL classes (currently only used by ComparisonCanvas) |

### Channel Text Parameters (All Canvas Classes)

These parameters control the channel identification text displayed on plots.

| Parameter | Type | Required/Default | Description | Notes |
|-----------|------|------------------|-------------|-------|
| `channel` | str | *Default: none* | Channel text to display | Supports ROOT LaTeX (e.g., `"e#mu channel"` → "eμ channel") |
| `channelPosX` | float | *Default: 0.2* | X position in NDC coordinates | Range: 0-1 (0=left, 1=right) |
| `channelPosY` | float | *Default: 0.7* | Y position in NDC coordinates | Range: 0-1 (0=bottom, 1=top) |
| `channelFont` | int | *Default: 61* | ROOT font code | 42=regular, 52=italic, 61=bold, 62=bold italic |
| `channelAlign` | int | *Default: 0* | Text alignment code | 11=left-top, 21=center-top, 31=right-top, etc. |
| `channelSize` | float | *Default: 0.05* | Text size | Typical range: 0.04-0.07 |

### ComparisonCanvas-Specific Parameters

Additional parameters specific to `ComparisonCanvas` (data vs MC comparison with ratio).

| Parameter | Type | Required/Default | Description | Notes |
|-----------|------|------------------|-------------|-------|
| `rTitle` | str | *Default: "Data / Pred"* | Ratio panel Y-axis title | |
| `rRange` | list | *Default: [0.5, 1.5]* | Ratio panel Y-axis range | `[rmin, rmax]` |

### KinematicCanvasWithRatio-Specific Parameters

Additional parameters specific to `KinematicCanvasWithRatio` (multi-sample comparison with ratio).

| Parameter | Type | Required/Default | Description | Notes |
|-----------|------|------------------|-------------|-------|
| `rTitle` | str | *Default: "Ratio"* | Ratio panel Y-axis title | |
| `rRange` | list | *Default: [0.5, 1.5]* | Ratio panel Y-axis range | `[rmin, rmax]` |

## Parameter Usage by Canvas Class

✓ = Available and functional
○ = Available but not currently used
✗ = Not applicable

| Parameter | ComparisonCanvas | KinematicCanvas | KinematicCanvasWithRatio |
|-----------|------------------|-----------------|--------------------------|
| **Basic Configuration** ||||
| `era` | ✓ | ✓ | ✓ |
| `CoM` | ✓ | ✓ | ✓ |
| `xTitle` | ✓ | ✓ | ✓ |
| `yTitle` | ✓ | ✓ | ✓ |
| `xRange` | ✓ | ✓ | ✓ |
| `yRange` | ✓ | ✓ | ✓ |
| **Binning** ||||
| `rebin` | ✓ | ✓ | ✓ |
| `overflow` | ✓ (NEW!) | ✓ | ✓ |
| **Display Options** ||||
| `logy` | ✓ | ✓ | ✓ |
| `normalize` | ✗ | ✓ | ✓ |
| `prescaled` | ✓ | ✓ | ✓ |
| `maxDigits` | ✓ | ✓ | ✓ |
| `legend` | ✓ | ✓ | ✓ |
| `systSrc` | ✓ | ○ | ○ |
| **Channel Text** ||||
| `channel` | ✓ | ✓ | ✓ |
| `channelPosX` | ✓ | ✓ | ✓ |
| `channelPosY` | ✓ | ✓ | ✓ |
| `channelFont` | ✓ | ✓ | ✓ |
| `channelAlign` | ✓ | ✓ | ✓ |
| `channelSize` | ✓ | ✓ | ✓ |
| **Ratio Panel** ||||
| `rTitle` | ✓ | ✗ | ✓ |
| `rRange` | ✓ | ✗ | ✓ |

## Detailed Parameter Descriptions

### xRange - Axis Range and Variable Binning

The `xRange` parameter serves dual purposes:

**Simple axis range (2 elements):**
```json
"xRange": [0, 200]
```
Sets axis from 0 to 200 GeV. No rebinning applied unless `rebin` is specified.

**Variable binning (>2 elements):**
```json
"xRange": [10, 15, 20, 30, 50, 75, 100, 200]
```
Creates bins with edges at specified values:
- Bin 1: 10-15 GeV (width: 5 GeV)
- Bin 2: 15-20 GeV (width: 5 GeV)
- Bin 3: 20-30 GeV (width: 10 GeV)
- etc.

When variable binning is used, the `rebin` parameter is ignored.

**⚠️ Important:** Bin edges should align with original histogram bin boundaries to avoid ROOT warnings.

### overflow - Overflow Bin Handling

**NEW FEATURE**: Now available in all three canvas classes (previously only in KinematicCanvas classes).

When `overflow: true` and the x-axis range is asymmetric:
- Accumulates all bins beyond `xRange[-1]` into the last visible bin
- Updates bin content and error appropriately
- Only applies to asymmetric ranges (where `abs(xmin) != abs(xmax)`)

**Example:**
```json
{
    "xRange": [0, 100],
    "overflow": true
}
```

This will accumulate all events above 100 GeV into the 100 GeV bin.

**When overflow is applied:**
- Asymmetric range like `[0, 100]` → overflow applied
- Symmetric range like `[-2.5, 2.5]` → overflow NOT applied

### systSrc - Systematics Source Label

**NEW**: Now available as a configuration parameter in all classes (previously hardcoded).

Default value: `"Stat+Syst"`

Used by ComparisonCanvas to label the systematics uncertainty band in the legend. Available in other classes for future use.

**Example:**
```json
{
    "systSrc": "MC Stat ⊕ Syst"
}
```

### normalize - Histogram Normalization

When `normalize: true`:
- Each histogram is scaled to unit area
- Integration includes overflow and underflow bins
- Useful for shape comparisons

**Not applicable to ComparisonCanvas** (data/MC comparison should preserve absolute normalization).

### channel - Channel Identification Text

Displays text identifying the analysis channel on the plot. Supports ROOT's LaTeX-like syntax.

**Common patterns:**
```python
"e#mu channel"          # eμ channel
"#mu#mu channel"        # μμ channel
"3#mu SR"              # 3μ SR
"1e2#mu"               # 1e2μ
"SS e^{#pm}#mu^{#pm}"  # SS e±μ±
"OS e^{#pm}#mu^{#mp}"  # OS e±μ∓
"#gamma#gamma"         # γγ
```

**ROOT LaTeX symbols:**
- `#mu` → μ
- `#tau` → τ
- `#nu` → ν
- `#gamma` → γ
- `#pm` → ±
- `#mp` → ∓
- `^{text}` → superscript
- `_{text}` → subscript

### channelFont - Font Codes

ROOT font codes:
- `42`: Helvetica regular (normal weight)
- `52`: Helvetica italic
- `61`: Helvetica bold (**default**)
- `62`: Helvetica bold italic
- `72`: Times regular
- `82`: Times bold

### channelAlign - Alignment Codes

Text alignment relative to `(channelPosX, channelPosY)`:

| Code | Horizontal | Vertical | Description |
|------|------------|----------|-------------|
| 11 | Left | Top | Left-top alignment |
| 12 | Left | Center | Left-center alignment |
| 13 | Left | Bottom | Left-bottom alignment |
| 21 | Center | Top | Center-top alignment |
| 22 | Center | Center | Center-center alignment |
| 23 | Center | Bottom | Center-bottom alignment |
| 31 | Right | Top | Right-top alignment |
| 32 | Right | Center | Right-center alignment |
| 33 | Right | Bottom | Right-bottom alignment |

**Default (0)**: No specific alignment (equivalent to left-bottom).

## Configuration Examples

### Example 1: Basic ComparisonCanvas (Data vs MC)

```json
{
    "era": "2022",
    "xRange": [0, 200],
    "xTitle": "p_{T} [GeV]",
    "yTitle": "Events / 5 GeV",
    "rebin": 5,
    "channel": "e#mu channel"
}
```

**Result:**
- Uses 2022 data (13.6 TeV, 7.9 fb⁻¹)
- X-axis: 0-200 GeV
- Rebins by factor of 5
- Shows "eμ channel" at default position (0.2, 0.7)

### Example 2: ComparisonCanvas with Overflow (NEW!)

```json
{
    "era": "2023",
    "xRange": [20, 150],
    "xTitle": "p_{T}(#mu) [GeV]",
    "yTitle": "Events",
    "overflow": true,
    "channel": "#mu#mu SR",
    "channelPosX": 0.25,
    "channelPosY": 0.75
}
```

**Result:**
- All events with pT > 150 GeV accumulated into 150 GeV bin
- Channel text "μμ SR" positioned at (0.25, 0.75)

### Example 3: Variable Binning with Fine Structure

```json
{
    "era": "Run2",
    "xRange": [60, 70, 80, 85, 88, 90, 91, 92, 93, 95, 97, 100, 110, 120],
    "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
    "yTitle": "Events / bin",
    "channel": "Z#rightarrow#mu#mu",
    "channelPosX": 0.72,
    "channelPosY": 0.82
}
```

**Result:**
- Fine binning (1 GeV) near Z peak (88-95 GeV)
- Coarse binning elsewhere
- No fixed rebinning factor needed

### Example 4: KinematicCanvas with Normalization

```json
{
    "era": "2022",
    "CoM": 13.6,
    "xRange": [0, 500],
    "xTitle": "H_{T} [GeV]",
    "yTitle": "Normalized",
    "rebin": 10,
    "normalize": true,
    "logy": true,
    "overflow": true,
    "channel": "Signal comparison"
}
```

**Result:**
- All histograms normalized to unit area
- Logarithmic Y-axis
- Overflow bins accumulated
- Useful for shape comparison

### Example 5: KinematicCanvasWithRatio - Custom Ratio Range

```json
{
    "era": "2023BPix",
    "xRange": [0, 200],
    "xTitle": "E_{T}^{miss} [GeV]",
    "yTitle": "Events",
    "rTitle": "Ratio to Nominal",
    "rRange": [0.8, 1.2],
    "rebin": 5,
    "channel": "VR",
    "channelPosX": 0.85,
    "channelPosY": 0.88,
    "channelAlign": 31
}
```

**Result:**
- Ratio panel shows 0.8-1.2 range
- Channel text "VR" in top-right corner (right-aligned)

### Example 6: Custom Systematics Label

```json
{
    "era": "2022EE",
    "xRange": [0, 300],
    "xTitle": "M_{3l} [GeV]",
    "yTitle": "Events / 10 GeV",
    "rebin": 10,
    "systSrc": "MC Stat#oplussyst",
    "channel": "1e2#mu SR"
}
```

**Result:**
- Custom systematics label in legend
- Channel text shows "1e2μ SR"

## Migration from Old Code

If you have existing configurations, they will continue to work unchanged (full backward compatibility).

**Old code:**
```python
config = {
    "era": "2022",
    "xRange": [0, 200],
    "xTitle": "p_{T} [GeV]"
    # Missing optional parameters
}
```

**Still works!** All optional parameters have sensible defaults.

**New features to consider adding:**
- `overflow: true` for ComparisonCanvas (now available!)
- `systSrc` to customize systematics label
- `channel` and related parameters for channel identification

## See Also

- `VARIABLE_BINNING.md` - Detailed guide on variable binning
- `CHANNEL_TEXT.md` - Detailed guide on channel text customization
- `example_channel_text_configs.json` - Example configurations
- `example_variable_binning_config.json` - Variable binning examples
- `test_channel_text.py` - Test suite demonstrating all features
- `test_variable_binning.py` - Variable binning test suite

## Notes

1. **Backward Compatibility**: All existing configurations will continue to work unchanged.

2. **Parameter Validation**: The code uses `config.get(key, default)` throughout, so missing parameters automatically use sensible defaults.

3. **Era Auto-Detection**: If `CoM` is not specified, it's automatically determined from the `era` parameter.

4. **Y-Range Auto-Calculation**: If `yRange` is not specified, it's automatically calculated from histogram content, with appropriate scaling for linear/log axes.

5. **Variable Binning Priority**: When both `rebin` and variable binning (xRange with >2 elements) are specified, variable binning takes precedence.

6. **Overflow Symmetry Check**: Overflow handling only applies to asymmetric ranges to avoid double-counting in symmetric distributions (e.g., η, φ).

7. **systSrc Future Use**: While `systSrc` is now available in all classes, it's currently only actively used by ComparisonCanvas. It may be utilized by other classes in future updates.
