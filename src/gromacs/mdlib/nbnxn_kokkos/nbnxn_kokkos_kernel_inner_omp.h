#if defined CHECK_EXCLS && (defined CALC_COULOMB || defined LJ_EWALD)
#define EXCL_FORCES
#endif

{
    const int cj = cj_[I](cjind).cj;
    const unsigned int excl = cj_[I](cjind).excl;

    Kokkos::parallel_for
        (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k)
         {
             int ajk = cj * UNROLLJ + k;

             xj(k,XX) = x_(ajk,XX);
             xj(k,YY) = x_(ajk,YY);
             xj(k,ZZ) = x_(ajk,ZZ);

             qj(k)    =  q_(ajk);

             fj(k,XX) = 0.0;
             fj(k,YY) = 0.0;
             fj(k,ZZ) = 0.0;

         });

    struct v_energy V_sum1;

    Kokkos::parallel_reduce
        (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k, struct v_energy& V)
         {
             int ai0         = ci*UNROLLI + 0;
             int ai1         = ai0 + 1;
             int type_i_off0 = type_(ai0)*ntype2;
             int type_i_off1 = type_(ai1)*ntype2;

             int aj         = cj*UNROLLJ + k;
             int type_j_off = type_(aj)*2;

             real c60       = nbfp_(type_i_off0 + type_j_off);
             real c120      = nbfp_(type_i_off0 + type_j_off + 1);
             real c61       = nbfp_(type_i_off1 + type_j_off);
             real c121      = nbfp_(type_i_off1 + type_j_off + 1);

             real dx0, dy0, dz0;
             real rsq0, rinv0;
             real rinvsq0, rinvsix0;
             real FrLJ60 = 0, FrLJ120 = 0, frLJ0 = 0, VLJ0 = 0;

             real dx1, dy1, dz1;
             real rsq1, rinv1;
             real rinvsq1, rinvsix1;
             real FrLJ61 = 0, FrLJ121 = 0, frLJ1 = 0, VLJ1 = 0;

#ifdef CALC_COULOMB
             real qq0;
             real fcoul0;
             real rs0, frac0;
             int  ri0;
             real fexcl0;
             real vcoul0;

             real qq1;
             real fcoul1;
             real rs1, frac1;
             int  ri1;
             real fexcl1;
             real vcoul1;
#endif
             real fscal0;
             real fx0, fy0, fz0;

             real fscal1;
             real fx1, fy1, fz1;

             /* A multiply mask used to zero an interaction
              * when either the distance cutoff is exceeded, or
              * (if appropriate) the i and j indices are
              * unsuitable for this kind of inner loop. */
             real skipmask0;
             real skipmask1;

#ifdef CHECK_EXCLS
             /* A multiply mask used to zero an interaction
              * when that interaction should be excluded
              * (e.g. because of bonding). */
             int interact0;
             int interact1;

             interact0 = ((excl>>(k)) & 1);
             interact1 = ((excl>>(UNROLLI + k)) & 1);

#ifndef EXCL_FORCES
             skipmask0 = interact0;
             skipmask1 = interact1;
#else
             skipmask0 = !(cj == ci_sh && k <= 0);
             skipmask1 = !(cj == ci_sh && k <= 1);
#endif
#else
#define interact0 1.0
#define interact1 1.0
             skipmask0 = 1.0;
             skipmask1 = 1.0;
#endif
             dx0  = xi(0,XX) - xj(k,XX);
             dy0  = xi(0,YY) - xj(k,YY);
             dz0  = xi(0,ZZ) - xj(k,ZZ);

             dx1  = xi(1,XX) - xj(k,XX);
             dy1  = xi(1,YY) - xj(k,YY);
             dz1  = xi(1,ZZ) - xj(k,ZZ);

             rsq0 = dx0*dx0 + dy0*dy0 + dz0*dz0;
             rsq1 = dx1*dx1 + dy1*dy1 + dz1*dz1;

             /* Prepare to enforce the cut-off. */
             skipmask0 = (rsq0 >= rcut2) ? 0 : skipmask0;
             skipmask1 = (rsq1 >= rcut2) ? 0 : skipmask1;

#ifdef CHECK_EXCLS
             /* Excluded atoms are allowed to be on top of each other.
              * To avoid overflow of rinv, rinvsq and rinvsix
              * we add a small number to rsq for excluded pairs only.
              */
             rsq0 += (1 - interact0)*NBNXN_AVOID_SING_R2_INC;
             rsq1 += (1 - interact1)*NBNXN_AVOID_SING_R2_INC;
#endif
             
             /* sqrt() function can be used in autovectorized loop */
             /* gmx_invsqrt() found to be slower than just 1/sqrt(rsq) */
             /* may be gmx_invsqrt is using table based gromacs approach which may be */
             /* inefficient for autovectorization */
             rinv0 = 1.0/sqrt(rsq0);
             rinv1 = 1.0/sqrt(rsq1);

             /* Partially enforce the cut-off (and perhaps
              * exclusions) to avoid possible overflow of
              * rinvsix when computing LJ, and/or overflowing
              * the Coulomb table during lookup. */
             rinv0 = rinv0 * skipmask0;
             rinv1 = rinv1 * skipmask1;

             rinvsq0  = rinv0*rinv0;
             rinvsq1  = rinv1*rinv1;

             rinvsix0 = interact0*rinvsq0*rinvsq0*rinvsq0;
             FrLJ60   = c60*rinvsix0;
             FrLJ120  = c120*rinvsix0*rinvsix0;
             frLJ0    = FrLJ120 - FrLJ60;
             VLJ0     = (FrLJ120 + c120*repulsion_shift_cpot_)/12 -
                 (FrLJ60 + c60*dispersion_shift_cpot_)/6;

             rinvsix1 = interact1*rinvsq1*rinvsq1*rinvsq1;
             FrLJ61   = c61*rinvsix1;
             FrLJ121  = c121*rinvsix1*rinvsix1;
             frLJ1    = FrLJ121 - FrLJ61;
             VLJ1     = (FrLJ121 + c121*repulsion_shift_cpot_)/12 -
                 (FrLJ61 + c61*dispersion_shift_cpot_)/6;

             /* Need to zero the interaction if there should be exclusion. */
             VLJ0     = VLJ0 * interact0;
             VLJ1     = VLJ1 * interact1;
             /* Need to zero the interaction if r >= rcut */
             VLJ0     = VLJ0 * skipmask0;
             VLJ1     = VLJ1 * skipmask1;

             /* Vvdw_ci += VLJ0 + VLJ1; */
             V.vdw += VLJ0 + VLJ1;
#ifdef CALC_COULOMB

             qq0     = skipmask0 * qi(0) * qj(k);
             rs0     = rsq0*rinv0*tabq_scale_;
             ri0     = (int)rs0;
             frac0   = rs0 - ri0;
             fexcl0  = (1 - frac0)*Ftab_(ri0) + frac0*Ftab_(ri0+1);
             fcoul0  = qq0*rinv0 * (interact0*rinvsq0 - fexcl0);
             vcoul0  = qq0*(interact0*(rinv0 - sh_ewald_)
                            -(Vtab_(ri0) - halfsp*frac0*(Ftab_(ri0) + fexcl0)));

             qq1     = skipmask1 * qi(1) * qj(k);
             rs1     = rsq1*rinv1*tabq_scale_;
             ri1     = (int)rs1;
             frac1   = rs1 - ri1;
             fexcl1  = (1 - frac1)*Ftab_(ri1) + frac1*Ftab_(ri1+1);
             fcoul1  = qq1*rinv1 * (interact1*rinvsq1 - fexcl1);
             vcoul1  = qq1*(interact1*(rinv1 - sh_ewald_)
                            -(Vtab_(ri1) - halfsp*frac1*(Ftab_(ri1) + fexcl1)));

             /* Vc_ci += vcoul0 + vcoul1; */
             V.vc += vcoul0 + vcoul1;
#endif

#ifdef CALC_COULOMB
             fscal0 = frLJ0*rinvsq0 + fcoul0;
             fscal1 = frLJ1*rinvsq1 + fcoul1;
#else
             fscal0 = frLJ0*rinvsq0;
             fscal1 = frLJ1*rinvsq1;
#endif
             fx0 = fscal0*dx0;
             fy0 = fscal0*dy0;
             fz0 = fscal0*dz0;

             fx1 = fscal1*dx1;
             fy1 = fscal1*dy1;
             fz1 = fscal1*dz1;

             fi(0,XX) += fx0;
             fi(0,YY) += fy0;
             fi(0,ZZ) += fz0;

             fi(1,XX) += fx1;
             fi(1,YY) += fy1;
             fi(1,ZZ) += fz1;

             fj(k,XX) += fx0 + fx1;
             fj(k,YY) += fy0 + fy1;
             fj(k,ZZ) += fz0 + fz1;

         }, V_sum1);

    struct v_energy V_sum2;
    Kokkos::parallel_reduce
        (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k, struct v_energy& V)
         {
             int ai0         = ci*UNROLLI + 2;
             int ai1         = ai0 + 1;
             int type_i_off0 = type_(ai0)*ntype2;
             int type_i_off1 = type_(ai1)*ntype2;

             int aj         = cj*UNROLLJ + k;
             int type_j_off = type_(aj)*2;

             real c60       = nbfp_(type_i_off0 + type_j_off);
             real c120      = nbfp_(type_i_off0 + type_j_off + 1);
             real c61       = nbfp_(type_i_off1 + type_j_off);
             real c121      = nbfp_(type_i_off1 + type_j_off + 1);

             real dx0, dy0, dz0;
             real rsq0, rinv0;
             real rinvsq0, rinvsix0;
             real FrLJ60 = 0, FrLJ120 = 0, frLJ0 = 0, VLJ0 = 0;

             real dx1, dy1, dz1;
             real rsq1, rinv1;
             real rinvsq1, rinvsix1;
             real FrLJ61 = 0, FrLJ121 = 0, frLJ1 = 0, VLJ1 = 0;

#ifdef CALC_COULOMB
             real qq0;
             real fcoul0;
             real rs0, frac0;
             int  ri0;
             real fexcl0;
             real vcoul0;

             real qq1;
             real fcoul1;
             real rs1, frac1;
             int  ri1;
             real fexcl1;
             real vcoul1;
#endif
             real fscal0 = 0.0;
             real fx0, fy0, fz0;

             real fscal1 = 0.0;
             real fx1, fy1, fz1;

             /* A multiply mask used to zero an interaction
              * when either the distance cutoff is exceeded, or
              * (if appropriate) the i and j indices are
              * unsuitable for this kind of inner loop. */
             real skipmask0;
             real skipmask1;

#ifdef CHECK_EXCLS
             /* A multiply mask used to zero an interaction
              * when that interaction should be excluded
              * (e.g. because of bonding). */
             int interact0;
             int interact1;

             interact0 = ((excl>>(2*UNROLLI + k)) & 1);
             interact1 = ((excl>>(3*UNROLLI + k)) & 1);

#ifndef EXCL_FORCES
             skipmask0 = interact0;
             skipmask1 = interact1;
#else
             skipmask0 = !(cj == ci_sh && k <= 2);
             skipmask1 = !(cj == ci_sh && k <= 3);
#endif
#else
#define interact0 1.0
#define interact1 1.0
             skipmask0 = 1.0;
             skipmask1 = 1.0;
#endif
             dx0  = xi(2,XX) - xj(k,XX);
             dy0  = xi(2,YY) - xj(k,YY);
             dz0  = xi(2,ZZ) - xj(k,ZZ);

             dx1  = xi(3,XX) - xj(k,XX);
             dy1  = xi(3,YY) - xj(k,YY);
             dz1  = xi(3,ZZ) - xj(k,ZZ);

             rsq0 = dx0*dx0 + dy0*dy0 + dz0*dz0;
             rsq1 = dx1*dx1 + dy1*dy1 + dz1*dz1;

             /* Prepare to enforce the cut-off. */
             skipmask0 = (rsq0 >= rcut2) ? 0 : skipmask0;
             skipmask1 = (rsq1 >= rcut2) ? 0 : skipmask1;

#ifdef CHECK_EXCLS
             /* Excluded atoms are allowed to be on top of each other.
              * To avoid overflow of rinv, rinvsq and rinvsix
              * we add a small number to rsq for excluded pairs only.
              */
             rsq0 += (1 - interact0)*NBNXN_AVOID_SING_R2_INC;
             rsq1 += (1 - interact1)*NBNXN_AVOID_SING_R2_INC;
#endif
             
             /* sqrt() function can be used in autovectorized loop */
             /* gmx_invsqrt() found to be slower than just 1/sqrt(rsq) */
             /* may be gmx_invsqrt is using table based gromacs approach which may be */
             /* inefficient for autovectorization */
             rinv0 = 1.0/sqrt(rsq0);
             rinv1 = 1.0/sqrt(rsq1);

             /* Partially enforce the cut-off (and perhaps
              * exclusions) to avoid possible overflow of
              * rinvsix when computing LJ, and/or overflowing
              * the Coulomb table during lookup. */
             rinv0 = rinv0 * skipmask0;
             rinv1 = rinv1 * skipmask1;

             rinvsq0  = rinv0*rinv0;
             rinvsq1  = rinv1*rinv1;

#ifndef HALF_LJ
             rinvsix0 = interact0*rinvsq0*rinvsq0*rinvsq0;
             FrLJ60   = c60*rinvsix0;
             FrLJ120  = c120*rinvsix0*rinvsix0;
             frLJ0    = FrLJ120 - FrLJ60;
             VLJ0     = (FrLJ120 + c120*repulsion_shift_cpot_)/12 -
                 (FrLJ60 + c60*dispersion_shift_cpot_)/6;

             rinvsix1 = interact1*rinvsq1*rinvsq1*rinvsq1;
             FrLJ61   = c61*rinvsix1;
             FrLJ121  = c121*rinvsix1*rinvsix1;
             frLJ1    = FrLJ121 - FrLJ61;
             VLJ1     = (FrLJ121 + c121*repulsion_shift_cpot_)/12 -
                 (FrLJ61 + c61*dispersion_shift_cpot_)/6;

             /* Need to zero the interaction if there should be exclusion. */
             VLJ0     = VLJ0 * interact0;
             VLJ1     = VLJ1 * interact1;
             /* Need to zero the interaction if r >= rcut */
             VLJ0     = VLJ0 * skipmask0;
             VLJ1     = VLJ1 * skipmask1;
             /* 1 more flop for LJ energy */

             /* Vvdw_ci += VLJ0 + VLJ1; */
             V.vdw += VLJ0 + VLJ1;
#endif

#ifdef CALC_COULOMB

             qq0     = skipmask0 * qi(2) * qj(k);
             rs0     = rsq0*rinv0*tabq_scale_;
             ri0     = (int)rs0;
             frac0   = rs0 - ri0;
             fexcl0  = (1 - frac0)*Ftab_(ri0) + frac0*Ftab_(ri0+1);
             fcoul0  = qq0*rinv0 * (interact0*rinvsq0 - fexcl0);
             vcoul0  = qq0*(interact0*(rinv0 - sh_ewald_)
                            -(Vtab_(ri0) - halfsp*frac0*(Ftab_(ri0) + fexcl0)));

             qq1     = skipmask1 * qi(3) * qj(k);
             rs1     = rsq1*rinv1*tabq_scale_;
             ri1     = (int)rs1;
             frac1   = rs1 - ri1;
             fexcl1  = (1 - frac1)*Ftab_(ri1) + frac1*Ftab_(ri1+1);
             fcoul1  = qq1*rinv1 * (interact1*rinvsq1 - fexcl1);
             vcoul1  = qq1*(interact1*(rinv1 - sh_ewald_)
                            -(Vtab_(ri1) - halfsp*frac1*(Ftab_(ri1) + fexcl1)));

             /* Vc_ci += vcoul0 + vcoul1; */
             V.vc += vcoul0 + vcoul1;
#endif

#ifndef HALF_LJ
             fscal0 = frLJ0*rinvsq0;
             fscal1 = frLJ1*rinvsq1;
#endif

#ifdef CALC_COULOMB
             fscal0 += fcoul0;
             fscal1 += fcoul1;
#endif
             fx0 = fscal0*dx0;
             fy0 = fscal0*dy0;
             fz0 = fscal0*dz0;

             fx1 = fscal1*dx1;
             fy1 = fscal1*dy1;
             fz1 = fscal1*dz1;

             fi(2,XX) += fx0;
             fi(2,YY) += fy0;
             fi(2,ZZ) += fz0;

             fi(3,XX) += fx1;
             fi(3,YY) += fy1;
             fi(3,ZZ) += fz1;

             fj(k,XX) += fx0 + fx1;
             fj(k,YY) += fy0 + fy1;
             fj(k,ZZ) += fz0 + fz1;

         }, V_sum2);

    Vvdw_ci += V_sum1.vdw + V_sum2.vdw;
    Vc_ci   += V_sum1.vc + V_sum2.vc;

    Kokkos::parallel_for
        (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k)
         {
             int ajk = cj * UNROLLJ + k;
             f_view(ajk, XX) -= fj(k,XX);
             f_view(ajk, YY) -= fj(k,YY);
             f_view(ajk, ZZ) -= fj(k,ZZ);
         });

}

#undef interact0
#undef interact1
#undef EXCL_FORCES
