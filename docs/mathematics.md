# Mathematics

All equations used in the pipeline, in the order they are applied.

---

## 1. Coordinate Conversion (DEM spacing)

The DEM (STAC fallback path) is loaded in geographic coordinates (degrees). Gradient computation requires metric spacing.

$$dx = \Delta\lambda \cdot 111320 \cdot \cos\!\left(\phi_0\right) \quad [\text{metres per cell}]$$

$$dy = \Delta\phi \cdot 111320 \quad [\text{metres per cell}]$$

where:
- $\Delta\lambda$ = longitude step size (degrees)
- $\Delta\phi$ = latitude step size (degrees)
- $\phi_0$ = latitude at the centre of the domain (radians)
- $111320$ m/deg = Earth's mean meridional arc length per degree

---

## 2. Terrain Derivatives

Computed on the GPU via finite differences (`numpy.gradient` / `cupy.gradient`) with the metric spacings $dx$ and $dy$. Only used in the STAC fallback path; the H3 Parquet DEM includes pre-computed derivatives.

### 2.1 Slope

Rate of change of elevation in the steepest direction:

$$\theta = \arctan\!\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right) \cdot \frac{180\degree}{\pi}$$

Units: degrees (0 deg = flat, 90 deg = vertical cliff).

### 2.2 Aspect

Compass direction of the steepest downslope, measured clockwise from north:

$$\psi = \left(\arctan2\!\left(-\frac{\partial z}{\partial x},\; -\frac{\partial z}{\partial y}\right) \cdot \frac{180\degree}{\pi} + 360\degree\right) \bmod 360\degree$$

Units: degrees (0 deg = north, 90 deg = east, 180 deg = south, 270 deg = west).

### 2.3 Terrain Ruggedness Index (TRI)

Root-mean-square deviation of elevation within a 3x3 kernel neighbourhood:

$$\text{TRI}(i,j) = \sqrt{\frac{1}{9}\sum_{k=-1}^{1}\sum_{l=-1}^{1} \left(z_{i+k,\,j+l} - \bar{z}_{3\times3}\right)^2}$$

Equivalent to: `sqrt(convolve((z - mean(z))^2, uniform_3x3_kernel))`.

### 2.4 Topographic Position Index (TPI)

Elevation relative to the mean of the surrounding 5x5 neighbourhood:

$$\text{TPI}(i,j) = z_{i,j} - \frac{1}{25}\sum_{k=-2}^{2}\sum_{l=-2}^{2} z_{i+k,\,j+l}$$

Positive TPI = ridgeline, negative TPI = valley.

---

## 3. H3 Hexagonal Grid

### 3.1 H3 cell area

At resolution $r$, each hexagonal cell has area:

$$A_r = A_0 \cdot 7^{-(r)}$$

where $A_0 \approx 4.25 \times 10^8$ km2 (area of a res-0 cell). In practice the library call `h3.cell_area(cell, 'km^2')` is used since cells are not perfectly regular on a sphere.

| Resolution | Approximate area |
|---|---|
| 5 | 253 km2 |
| 7 | 5.16 km2 |
| 9 | 0.105 km2 |

---

## 4. Bilinear Interpolation

### 4.1 The problem

Given $S = N_y \times N_x$ source values $f_{t,i,j}$ on a regular latitude-longitude grid with spacing $(\Delta\phi, \Delta\lambda)$, estimate $f$ at $N$ target H3 cell centres $\mathbf{p}_k = (\phi_k, \lambda_k)$, for each of $T$ timesteps.

### 4.2 Fractional grid indices

Each target point is mapped to fractional indices on the source grid:

$$i_k = \frac{\phi_k - \phi_0}{\Delta\phi}, \quad j_k = \frac{\lambda_k' - \lambda_0}{\Delta\lambda}$$

where:
- $\phi_0, \lambda_0$ = origin (first latitude, first longitude) of the source grid
- $\lambda_k'$ = target longitude normalised to the source grid range: $\lambda_k' = (\lambda_k - \lambda_0) \bmod L + \lambda_0$, where $L = \lambda_{\max} - \lambda_0 + \Delta\lambda$ is the full periodic range

### 4.3 Bilinear blend

For fractional indices $(i, j)$ where $\lfloor i \rfloor = i_0$ and $\lfloor j \rfloor = j_0$:

$$\hat{f}(\mathbf{p}_k) = (1-\alpha)(1-\beta)\,f_{i_0,j_0} + \alpha(1-\beta)\,f_{i_0+1,j_0} + (1-\alpha)\beta\,f_{i_0,j_0+1} + \alpha\beta\,f_{i_0+1,j_0+1}$$

where $\alpha = i - i_0$ and $\beta = j - j_0$ are the fractional parts.

This is equivalent to two successive linear interpolations (first along one axis, then the other) and is implemented via CuPy's `map_coordinates(data, coords, order=1)`.

### 4.4 Boundary handling

**Longitude wrap-around**: The source grid is circularly padded by 3 columns from each edge before interpolation. Target longitude indices are shifted by the pad width. This ensures correct interpolation near the 0/360 deg or -180/180 deg boundary without discontinuities.

**Latitude clamping**: Latitude indices are clamped to $[0, N_y - 1]$ (no wrap at poles). `mode='nearest'` in `map_coordinates` handles any remaining edge cases at the poles.

**NaN fallback**: After GPU bilinear interpolation, any remaining NaN values (rare edge cases at poles) are filled using scipy `RegularGridInterpolator` with `method='nearest'` on CPU.

### 4.5 Complexity

Each timestep processes all $N$ target points in a single GPU kernel call. Total complexity: $O(N \cdot T)$ -- independent of source grid size $S$. For the NOAA 0.25-degree grid (721 x 1440 = 1,038,240 source points) and H3 res 5 (2,016,842 targets), the full interpolation of 10+ variables across 21 timesteps completes in approximately 10-30 seconds on an A10G GPU.

### 4.6 CPU fallback

When no GPU is present, `scipy.interpolate.RegularGridInterpolator` with `method='linear'` provides equivalent bilinear interpolation. A second pass with `method='nearest'` fills any NaN values.

### 4.7 Note on Gaussian kernel smoothing (RBF)

For **regional** bounding boxes with few source grid points, Gaussian kernel smoothing (Nadaraya-Watson kernel regression) can produce smoother results by computing a weighted average over all source points. The Gaussian RBF kernel $\varphi(r) = \exp(-(r/\varepsilon)^2)$ with bandwidth $\varepsilon$ provides tunable smoothness. However, the O(N * S) complexity makes it prohibitively expensive for global runs (2 trillion distance calculations per timestep at res 5). Bilinear interpolation is both the standard method for regular-grid weather data and computationally tractable at global scale.

---

## 5. Topographic Corrections

After interpolation, the values at each H3 cell centre are corrected to account for the elevation difference between the coarse model grid elevation and the actual terrain elevation at that cell.

Let $\Delta z = z_\text{target} - z_\text{ref}$ where $z_\text{ref}$ is the mean DEM elevation of the configured bounding box, computed at runtime.

### 5.1 Temperature -- Variable Lapse Rate

Accounts for the decrease of temperature with altitude using a lapse rate derived from the model's own temperature profile each timestep (Karger et al. 2023, CHELSA-W5E5 approach):

$$\Gamma_t = \frac{\overline{T_{850,t} - T_{2\text{m},t}}}{z_{850} - z_\text{ref}} \quad [\text{deg C/m}]$$

$$T_\text{corrected} = T_\text{interp} + \Gamma_t \cdot \Delta z$$

where:
- $\overline{\cdot}$ = spatial mean over all target points at timestep $t$
- $z_{850} \approx 1500$ m MSL (geopotential height of 850 hPa)
- $z_\text{ref}$ = mean DEM elevation of the bounding box (computed at runtime)
- $\Gamma_t$ is clamped to $[-9.8, +5.0]$ deg C/km (dry adiabatic to strong inversion)

Falls back to fixed $\Gamma = -6.5$ deg C/km when T_850 is not available.

**References:** Karger et al. (2023), *ESSD*, doi:10.5194/essd-15-2445-2023; Dutra et al. (2020), *Earth & Space Science*, doi:10.1029/2019EA000984.

### 5.2 Wind Speed -- Elevation Enhancement

Wind speed increases with elevation as surface friction decreases:

$$V_\text{corrected} = V_\text{interp} \cdot \max\!\left(E_\text{base} \cdot E_\text{slope},\; 0.1\right)$$

$$E_\text{base} = 1 + \alpha \cdot \frac{\Delta z}{1000}$$

$$E_\text{slope} = 1 + \min\!\left(\frac{\theta}{30\degree},\; 1\right) \cdot 0.3$$

where:
- $\alpha = 0.3$ (wind height factor, dimensionless)
- The floor of 0.1 prevents unphysical near-zero wind in valleys

### 5.3 Precipitation -- Dynamic Orographic Enhancement

Precipitation increases on elevated, windward terrain as moist air is forced to rise (orographic lifting). The windward direction is computed dynamically from the model's u10/v10 fields at each timestep:

$$P_\text{corrected} = P_\text{interp} \cdot E_\text{base} \cdot E_\text{windward}(t) \cdot E_\text{slope}$$

$$E_\text{base} = 1 + \max\!\left(\frac{\Delta z}{500\;\text{m}},\; 0\right) \cdot 0.15$$

$$d_t = \left(\arctan2(-u_{10,t},\; -v_{10,t}) \cdot \frac{180\degree}{\pi} + 360\degree\right) \bmod 360\degree$$

$$E_\text{windward}(t) = 1 + \left(1 - \frac{|\psi - d_t|_\text{circ}}{180\degree}\right) \cdot 0.5$$

$$E_\text{slope} = 1 + \min\!\left(\frac{\theta}{45\degree},\; 1\right) \cdot 0.3$$

where:
- $d_t$ = meteorological wind direction ("from") at timestep $t$, computed per H3 cell
- $|\psi - d_t|_\text{circ} = \min(|\psi - d_t|,\; 360\degree - |\psi - d_t|)$ (circular angular difference)
- $E_\text{windward}$ reaches maximum 1.5 on slopes facing into the wind (windward) and minimum 1.0 on lee slopes
- Falls back to neutral factor (no directional enhancement) when u10/v10 are unavailable

**References:** Karger et al. (2017), *Scientific Data*, doi:10.1038/sdata2017122 (CHELSA dynamic orographic algorithm).

### 5.4 Humidity -- Exponential Elevation Correction

Specific humidity decreases exponentially with altitude following the atmospheric moisture scale-height profile:

$$q_\text{corrected} = q_\text{interp} \cdot \text{clip}\!\left(\exp\!\left(-\frac{\Delta z}{H_q}\right),\; 0.05,\; 1.5\right)$$

where $H_q = 2000$ m is the moisture scale height.

| $\Delta z$ | Correction factor |
|---|---|
| +500 m | 0.78 (-22%) |
| +1000 m | 0.61 (-39%) |
| +2000 m | 0.37 (-63%) |
| -500 m | 1.28 (+28%) |

**References:** Held & Soden (2006), *J. Climate*, doi:10.1175/JCLI3990.1; Trenberth et al. (2005), *J. Hydrometeorol.*

### 5.5 Upper-Air Variables -- No Terrain Correction

Variables on pressure levels (850 hPa temperature, 850 hPa winds, 500 hPa geopotential, 500 hPa vertical velocity) receive **no** topographic correction. These are free-atmosphere variables decoupled from the surface boundary layer. Published downscaling systems (CHELSA-W5E5, TopoSCALE) exclusively apply terrain corrections to surface variables.

**References:** Fiddes & Gruber (2014), *GMD*, doi:10.5194/gmd-7-387-2014.

---

## 6. Derived Variables

Computed from the corrected primary variables.

### 6.1 Wind speed and direction (surface 10 m)

$$V_{10} = \sqrt{u_{10}^2 + v_{10}^2}$$

$$d_{10} = \left(\arctan2(-u_{10},\; -v_{10}) \cdot \frac{180\degree}{\pi} + 360\degree\right) \bmod 360\degree$$

The negative signs in arctan2 convert from the meteorological convention (direction *from* which wind blows) correctly. $d=0\degree$ = wind from north, $d=90\degree$ = wind from east.

### 6.2 850 hPa wind speed and direction

$$V_{850} = \sqrt{u_{850}^2 + v_{850}^2}$$

$$d_{850} = \left(\arctan2(-u_{850},\; -v_{850}) \cdot \frac{180\degree}{\pi} + 360\degree\right) \bmod 360\degree$$

### 6.3 Wind shear (850 hPa -- 10 m)

Bulk wind shear between the boundary layer and the lower free troposphere. High shear is associated with convective organisation and severe weather.

$$\Delta u = u_{850} - u_{10}, \quad \Delta v = v_{850} - v_{10}$$

$$\text{shear\_magnitude} = \sqrt{\Delta u^2 + \Delta v^2}$$

$$\text{shear\_direction} = \left(\arctan2(-\Delta u,\; -\Delta v) \cdot \frac{180\degree}{\pi} + 360\degree\right) \bmod 360\degree$$

### 6.4 Temperature difference (atmospheric stability)

$$\Delta T = T_{850} - T_{2\text{m}}$$

A strongly negative $\Delta T$ (cold air above warm surface) indicates a conditionally unstable atmosphere prone to convection.

### 6.5 Moisture flux

Horizontal transport of water vapour -- a key predictor of heavy precipitation events:

$$\mathbf{Q} = q \cdot \mathbf{V} = (q\,u_{10},\; q\,v_{10})$$

$$|\mathbf{Q}| = \sqrt{(q\,u_{10})^2 + (q\,v_{10})^2}$$

Units: $\text{g kg}^{-1} \cdot \text{m s}^{-1}$

### 6.6 Geopotential height and anomaly

The geopotential $\Phi$ (m2 s-2) output by the model is converted to geopotential height:

$$H = \frac{\Phi}{g_0}, \quad g_0 = 9.80665\;\text{m s}^{-2}$$

The anomaly relative to the ICAO Standard Atmosphere 500 hPa height:

$$\Delta H_{500} = H_{500} - H_\text{ref}$$

where $H_\text{ref} = 5574$ m (configurable via `GEOPOTENTIAL_500_REF` in `config.py`).

Positive anomaly = anomalously high pressure ridge; negative = trough. Used for synoptic-scale pattern identification.

---

## 7. Unit Conversions

| Variable | Model output | Pipeline output | Conversion |
|---|---|---|---|
| Temperature | K | deg C | $T_C = T_K - 273.15$ |
| Pressure | Pa | hPa | $p_{hPa} = p_{Pa} / 100$ |
| Precipitation (6 h accum.) | m | mm | $P_{mm} = P_m \times 1000$ |
| Geopotential | m2 s-2 | m | $H = \Phi / g_0$ |
| Specific humidity | kg/kg | g/kg | $q_{g} = q_{kg} \times 1000$ |

---

## 8. Notation Summary

| Symbol | Meaning |
|---|---|
| $z$ | Elevation (m) |
| $\theta$ | Slope angle (degrees) |
| $\psi$ | Aspect angle (degrees from north) |
| $\Delta z$ | Target elevation minus reference elevation (m) |
| $\phi$ | Latitude (radians unless stated) |
| $\lambda$ | Longitude (radians unless stated) |
| $u, v$ | Zonal and meridional wind components (m/s) |
| $q$ | Specific humidity (g/kg) |
| $\omega$ | Vertical velocity (Pa/s); negative = upward motion |
| $\Phi$ | Geopotential (m2 s-2) |
| $g_0$ | Standard gravity 9.80665 m s-2 |
| $\Gamma$ | Environmental lapse rate (deg C/km) |
| $\alpha$ | Wind height factor 0.3 |
