// Typical room-temperature properties. Real values vary with alloy, temper, print settings
// and supplier - use "Custom" to enter your own datasheet numbers.
//   E [GPa], nu [-], yield / uts [MPa], density [kg/m^3]
//   brittle: failure is judged by max principal stress instead of von Mises
//   elongation: strain at break (plastic hardening from yield to uts, nonlinear study)
//   k [W/m K], cp [J/kg K], alpha [1/K x 1e-6]: thermal conductivity, heat capacity, expansion
//   fatigue: Se [MPa] fatigue strength at Ne cycles (fully reversed, polished); endurance: true
//            when the material has an endurance limit (no failure below Se); metal: surface
//            finish factors apply
//   cost: rough raw-material price [USD/kg] for comparing options, not a quote
export const MATERIALS = [
  { id: 'steel-1020', name: 'Steel, AISI 1020', E: 200, nu: 0.29, yield: 351, uts: 420, density: 7900, elongation: 0.25, k: 51.9, cp: 486, alpha: 11.7, fatigue: { Se: 210, Ne: 1e6, endurance: true, metal: true }, cost: 1.5 },
  { id: 'steel-alloy', name: 'Alloy steel (4140 Q&T)', E: 205, nu: 0.29, yield: 655, uts: 1020, density: 7850, elongation: 0.18, k: 42.6, cp: 473, alpha: 12.3, fatigue: { Se: 510, Ne: 1e6, endurance: true, metal: true }, cost: 2.5 },
  { id: 'ss-304', name: 'Stainless steel 304', E: 193, nu: 0.29, yield: 215, uts: 505, density: 8000, elongation: 0.45, k: 16.2, cp: 500, alpha: 17.3, fatigue: { Se: 240, Ne: 1e7, endurance: true, metal: true }, cost: 4 },
  { id: 'al-6061', name: 'Aluminium 6061-T6', E: 69, nu: 0.33, yield: 276, uts: 310, density: 2700, elongation: 0.12, k: 167, cp: 896, alpha: 23.6, fatigue: { Se: 96.5, Ne: 5e8, endurance: false, metal: true }, cost: 4 },
  { id: 'al-7075', name: 'Aluminium 7075-T6', E: 71.7, nu: 0.33, yield: 503, uts: 572, density: 2810, elongation: 0.11, k: 130, cp: 960, alpha: 23.6, fatigue: { Se: 159, Ne: 5e8, endurance: false, metal: true }, cost: 8 },
  { id: 'ti-64', name: 'Titanium Ti-6Al-4V', E: 114, nu: 0.34, yield: 880, uts: 950, density: 4430, elongation: 0.14, k: 6.7, cp: 526, alpha: 8.6, fatigue: { Se: 510, Ne: 1e7, endurance: true, metal: true }, cost: 45 },
  { id: 'brass', name: 'Brass C360', E: 97, nu: 0.31, yield: 310, uts: 385, density: 8500, elongation: 0.25, k: 115, cp: 380, alpha: 20.5, fatigue: { Se: 138, Ne: 1e8, endurance: false, metal: true }, cost: 8 },
  { id: 'copper', name: 'Copper C110 (annealed)', E: 115, nu: 0.33, yield: 69, uts: 220, density: 8900, elongation: 0.45, k: 388, cp: 385, alpha: 17, fatigue: { Se: 76, Ne: 1e8, endurance: false, metal: true }, cost: 10 },
  { id: 'cast-iron', name: 'Grey cast iron', E: 110, nu: 0.26, yield: 150, uts: 150, density: 7200, brittle: true, elongation: 0.005, k: 46, cp: 490, alpha: 11, fatigue: { Se: 69, Ne: 1e7, endurance: true, metal: true }, cost: 1.2 },
  { id: 'pla', name: 'PLA (3D printed, 100% infill)', E: 3.5, nu: 0.36, yield: 45, uts: 50, density: 1240, brittle: true, elongation: 0.05, k: 0.13, cp: 1800, alpha: 68, fatigue: { Se: 15, Ne: 1e7, endurance: false }, cost: 20 },
  { id: 'petg', name: 'PETG (3D printed)', E: 2.1, nu: 0.38, yield: 47, uts: 50, density: 1270, elongation: 0.1, k: 0.2, cp: 1200, alpha: 60, fatigue: { Se: 15, Ne: 1e7, endurance: false }, cost: 22 },
  { id: 'abs', name: 'ABS', E: 2.2, nu: 0.35, yield: 40, uts: 44, density: 1050, elongation: 0.1, k: 0.17, cp: 1400, alpha: 90, fatigue: { Se: 11, Ne: 1e7, endurance: false }, cost: 20 },
  { id: 'nylon', name: 'Nylon PA6/66', E: 2.8, nu: 0.39, yield: 70, uts: 80, density: 1140, elongation: 0.4, k: 0.25, cp: 1700, alpha: 90, fatigue: { Se: 25, Ne: 1e7, endurance: false }, cost: 30 },
  { id: 'pc', name: 'Polycarbonate', E: 2.4, nu: 0.37, yield: 62, uts: 70, density: 1200, elongation: 0.6, k: 0.2, cp: 1200, alpha: 68, fatigue: { Se: 14, Ne: 1e7, endurance: false }, cost: 35 },
  { id: 'cfrp', name: 'Carbon fibre (quasi-isotropic)', E: 60, nu: 0.3, yield: 570, uts: 570, density: 1600, brittle: true, elongation: 0.01, k: 5, cp: 900, alpha: 2, fatigue: { Se: 200, Ne: 1e7, endurance: false }, cost: 60 },
  { id: 'wood', name: 'Pine wood (along grain, approx.)', E: 9, nu: 0.3, yield: 40, uts: 40, density: 500, brittle: true, elongation: 0.01, k: 0.12, cp: 1700, alpha: 5, fatigue: { Se: 12, Ne: 1e7, endurance: false }, cost: 1 },
  { id: 'glass', name: 'Soda-lime glass', E: 70, nu: 0.22, yield: 45, uts: 45, density: 2500, brittle: true, elongation: 0.001, k: 1, cp: 840, alpha: 9, fatigue: { Se: 15, Ne: 1e7, endurance: false }, cost: 1.5 },
];

/** Fill in defaults for properties a custom or older material may lack. */
export function completeMaterial(m) {
  const out = { ...m };
  if (!(out.elongation > 0)) out.elongation = out.brittle ? 0.01 : 0.15;
  if (!(out.k > 0)) out.k = 50;
  if (!(out.cp > 0)) out.cp = 500;
  if (!Number.isFinite(out.alpha)) out.alpha = 12;
  if (!out.fatigue || !(out.fatigue.Se > 0)) out.fatigue = { Se: 0.4 * out.uts, Ne: 1e7, endurance: false, metal: false };
  if (!(out.cost >= 0)) out.cost = 5;
  return out;
}

export const UNITS = {
  mm: { label: 'mm', toMeters: 0.001 },
  cm: { label: 'cm', toMeters: 0.01 },
  m: { label: 'm', toMeters: 1 },
  in: { label: 'in', toMeters: 0.0254 },
};
