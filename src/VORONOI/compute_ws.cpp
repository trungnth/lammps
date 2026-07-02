// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Trung Nguyen at Dalat Nuclear Research Institute
------------------------------------------------------------------------- */

#include "compute_ws.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "update.h"

#include <cmath>
#include <cstring>

#include <voro++.hh>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeWS::ComputeWS(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), ws_array(nullptr), ref_x(nullptr), ref_tag(nullptr),
  ref_type(nullptr), assigned_site(nullptr), site_occupancy(nullptr), v_array(nullptr)
{
  if (narg != 3) error->all(FLERR, "Illegal compute ws command");

  peratom_flag = 1;
  size_peratom_cols = 4;
  nmax = 0;

  local_flag = 1;
  size_local_cols = 5; 
  n_vacancies = 0;

  vector_flag = 1;
  size_vector = 4;  
  extvector = 1;
  memory->create(vector, size_vector, "ws:vector");

  nref_max = 0;
  nref = 0;
  ref_stored = false;
  
  max_nall = 0;
}

/* ---------------------------------------------------------------------- */

ComputeWS::~ComputeWS()
{
  memory->destroy(ws_array);
  memory->destroy(ref_x);
  memory->destroy(ref_tag);
  memory->destroy(ref_type);
  memory->destroy(assigned_site);
  memory->destroy(site_occupancy);
  memory->destroy(v_array);
  memory->destroy(vector);
}

/* ---------------------------------------------------------------------- */

void ComputeWS::init()
{
  if (domain->dimension != 3) error->all(FLERR, "Compute ws requires 3D simulation");
}

/* ---------------------------------------------------------------------- */

void ComputeWS::store_reference()
{
  int nall = atom->nlocal + atom->nghost;

  if (nall > nref_max) {
    nref_max = atom->nmax; 
    memory->destroy(ref_x);
    memory->destroy(ref_tag);
    memory->destroy(ref_type);
    memory->destroy(site_occupancy);
    
    memory->create(ref_x, nref_max, 3, "ws:ref_x");
    memory->create(ref_tag, nref_max, "ws:ref_tag");
    memory->create(ref_type, nref_max, "ws:ref_type");
    memory->create(site_occupancy, nref_max, "ws:site_occupancy");
  }

  nref = nall;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *type = atom->type;

  for (int i = 0; i < nall; i++) {
    ref_x[i][0] = x[i][0];
    ref_x[i][1] = x[i][1];
    ref_x[i][2] = x[i][2];
    ref_tag[i] = tag[i];
    ref_type[i] = type[i];
  }
  ref_stored = true;
}

/* ---------------------------------------------------------------------- */

void ComputeWS::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  if (!ref_stored) store_reference();

  int nlocal = atom->nlocal;
  if (nlocal > nmax) {
    nmax = atom->nmax;
    memory->destroy(ws_array);
    memory->create(ws_array, nmax, 4, "ws:ws_array");
    array_atom = ws_array;
  }

  int nall = atom->nlocal + atom->nghost;
  
  if (nall > max_nall) {
    max_nall = atom->nmax; 
    memory->destroy(assigned_site);
    memory->create(assigned_site, max_nall, "ws:assigned_site");
  }

  memset(site_occupancy, 0, nref * sizeof(int));

  double *sublo = domain->sublo;
  double *subhi = domain->subhi;
  double skin = 10.0;

  double dx = subhi[0] - sublo[0] + 2.0 * skin;
  double dy = subhi[1] - sublo[1] + 2.0 * skin;
  double dz = subhi[2] - sublo[2] + 2.0 * skin;
  double vol = dx * dy * dz;

  double optimal_bins = nref / 5.0;
  if (optimal_bins < 1.0) optimal_bins = 1.0;

  double scale = pow(optimal_bins / vol, 1.0 / 3.0);
  int nx = round(dx * scale);
  int ny = round(dy * scale);
  int nz = round(dz * scale);

  if (nx < 1) nx = 1;
  if (ny < 1) ny = 1;
  if (nz < 1) nz = 1;

  voro::container con(sublo[0] - skin, subhi[0] + skin, sublo[1] - skin, subhi[1] + skin,
                      sublo[2] - skin, subhi[2] + skin, nx, ny, nz, false, false, false, 8);

  for (int i = 0; i < nref; i++) {
    con.put(i, ref_x[i][0], ref_x[i][1], ref_x[i][2]);
  }

  double **x = atom->x;

  for (int i = 0; i < nall; i++) {
    double rx, ry, rz;
    int site_id;
    if (con.find_voronoi_cell(x[i][0], x[i][1], x[i][2], rx, ry, rz, site_id)) {
      assigned_site[i] = site_id;
      if (site_id >= 0 && site_id < nref) {
        site_occupancy[site_id]++;
      }
    } else {
      assigned_site[i] = -1;
    }
  }

  tagint *tag = atom->tag;
  int *type = atom->type;

  int loc_vac = 0;
  int loc_rep = 0;
  int loc_anti = 0;
  int total_loc_int = 0;

  for (int i = 0; i < nlocal; i++) {
    int site = assigned_site[i];
    if (site >= 0 && site < nref) {
      ws_array[i][0] = site_occupancy[site];
      ws_array[i][1] = site + 1;
      ws_array[i][2] = ref_tag[site];
      ws_array[i][3] = ref_type[site];

      if (tag[i] != ref_tag[site]) loc_rep++;
      if (type[i] != ref_type[site]) loc_anti++;
    } else {
      ws_array[i][0] = 0.0;
      ws_array[i][1] = 0.0;
      ws_array[i][2] = 0.0;
      ws_array[i][3] = 0.0;
    }
  }

  n_vacancies = 0;
  for (int site = 0; site < nref; site++) {
    if (ref_x[site][0] >= sublo[0] && ref_x[site][0] < subhi[0] && 
        ref_x[site][1] >= sublo[1] && ref_x[site][1] < subhi[1] && 
        ref_x[site][2] >= sublo[2] && ref_x[site][2] < subhi[2]) {
      
      int occ = site_occupancy[site];
      
      if (occ > 1) {
        total_loc_int += (occ - 1);
      }
      if (occ == 0) {
        n_vacancies++;
        loc_vac++;
      }
    }
  }

  memory->destroy(v_array);
  memory->create(v_array, n_vacancies, 5, "ws:v_array");

  int v_idx = 0;
  for (int site = 0; site < nref; site++) {
    if (ref_x[site][0] >= sublo[0] && ref_x[site][0] < subhi[0] && 
        ref_x[site][1] >= sublo[1] && ref_x[site][1] < subhi[1] && 
        ref_x[site][2] >= sublo[2] && ref_x[site][2] < subhi[2]) {
      
      if (site_occupancy[site] == 0) {
        v_array[v_idx][0] = ref_tag[site];
        v_array[v_idx][1] = ref_type[site];
        v_array[v_idx][2] = ref_x[site][0];
        v_array[v_idx][3] = ref_x[site][1];
        v_array[v_idx][4] = ref_x[site][2];
        v_idx++;
      }
    }
  }

  array_local = v_array;
  size_local_rows = n_vacancies;

  double loc_stats[4] = {(double) total_loc_int, (double) loc_vac, (double) loc_rep,
                         (double) loc_anti};
  MPI_Allreduce(loc_stats, vector, 4, MPI_DOUBLE, MPI_SUM, world);
}

/* ---------------------------------------------------------------------- */

void ComputeWS::compute_local()
{
  invoked_local = update->ntimestep;
  if (invoked_peratom != update->ntimestep) compute_peratom();
}

/* ---------------------------------------------------------------------- */

void ComputeWS::compute_vector()
{
  invoked_vector = update->ntimestep;
  if (invoked_peratom != update->ntimestep) compute_peratom();
}

/* ---------------------------------------------------------------------- */

double ComputeWS::memory_usage()
{
  double bytes = 0.0;
  bytes += (double) nmax * 4 * sizeof(double); 
  bytes += (double) nref_max * 3 * sizeof(double); 
  bytes += (double) nref_max * sizeof(tagint); 
  bytes += (double) nref_max * sizeof(int); 
  bytes += (double) nref_max * sizeof(int); 
  bytes += (double) max_nall * sizeof(int); 
  bytes += (double) n_vacancies * 5 * sizeof(double); 
  return bytes;
}