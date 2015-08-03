#if defined CHECK_EXCLS && (defined CALC_COULOMB || defined LJ_EWALD)
#define EXCL_FORCES
#endif

{
    const int cj = cj_[I](cjind).cj;
    const unsigned int excl = cj_[I](cjind).excl;

    int cjj         = cj*UNROLLJ;
    Kokkos::parallel_for
        (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k)
         {
             int ajk = cjj + k;

             xj(k,XX) = x_(ajk,XX);
             xj(k,YY) = x_(ajk,YY);
             xj(k,ZZ) = x_(ajk,ZZ);

             qj(k)    =  q_(ajk);

             fj(k,XX) = 0.0;
             fj(k,YY) = 0.0;
             fj(k,ZZ) = 0.0;

         });

    // DOES NOT VECTORIZE
    Kokkos::parallel_for
        (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k)
         {
             int ajk = cjj + k;
             typej(k) = type_(ajk)*2;
         });

    for (int i = 0; i < 2; i++)
    {
        int type_i_off  = typei(i);
        real q          = qi(i);

        struct v_energy V_sum;
        Kokkos::parallel_reduce
            (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k,struct v_energy& V)
             {
                 int type_j_off = typej(k);

                 // reading lj parameter causes indirect access
                 real c6       = nbfp_(type_i_off + type_j_off);
                 real c12      = nbfp_(type_i_off + type_j_off + 1);

                 real dx, dy, dz;
                 real rsq, rinv;
                 real rinvsq, rinvsix;
                 real FrLJ6 = 0, FrLJ12 = 0, frLJ = 0, VLJ = 0;

#ifdef CALC_COULOMB
                 real qq;
                 real fcoul;
                 real rs, frac;
                 int  ri;
                 real fexcl;
                 real vcoul;
#endif
                 real fscal;
                 real fx, fy, fz;

                 real skipmask;

#ifdef CHECK_EXCLS
                 /* A multiply mask used to zero an interaction
                  * when that interaction should be excluded
                  * (e.g. because of bonding). */
                 int interact;

                 interact = ((excl>>(i*UNROLLI + k)) & 1);

#ifndef EXCL_FORCES
                 skipmask = interact;
#else
                 skipmask = !(cj == ci_sh && k <= i);
#endif
#else
#define interact 1.0
                 skipmask = 1.0;
#endif
                 dx  = xi(i,XX) - xj(k,XX);
                 dy  = xi(i,YY) - xj(k,YY);
                 dz  = xi(i,ZZ) - xj(k,ZZ);

                 rsq = dx*dx + dy*dy + dz*dz;

                 // prevents vectorization (complains loop with early exit)
                 /* /\* Prepare to enforce the cut-off. *\/ */
                 skipmask = (rsq >= rcut2) ? 0 : skipmask;

#ifdef CHECK_EXCLS
                 /* Excluded atoms are allowed to be on top of each other.
                  * To avoid overflow of rinv, rinvsq and rinvsix
                  * we add a small number to rsq for excluded pairs only.
                  */
                 rsq += (1 - interact)*NBNXN_AVOID_SING_R2_INC;
#endif
                 rinv = 1.0/sqrt(rsq);

                 rinv = rinv * skipmask;

                 rinvsq  = rinv*rinv;

                 rinvsix = interact*rinvsq*rinvsq*rinvsq;
                 FrLJ6   = c6*rinvsix;
                 FrLJ12  = c12*rinvsix*rinvsix;
                 frLJ    = FrLJ12 - FrLJ6;
                 VLJ     = (FrLJ12 + c12*repulsion_shift_cpot_)/12 -
                     (FrLJ6 + c6*dispersion_shift_cpot_)/6;

                 /* Need to zero the interaction if there should be exclusion. */
                 VLJ     = VLJ * interact;
                 /* Need to zero the interaction if r >= rcut */
                 VLJ     = VLJ * skipmask;

                 V.vdw +=  VLJ;

#ifdef CALC_COULOMB
                 qq     = skipmask * q * qj(k);
                 rs     = rsq*rinv*tabq_scale_;
                 ri     = (int)rs;
                 frac   = rs - ri;
                 fexcl  = (1 - frac)*Ftab_(ri) + frac*Ftab_(ri+1);
                 fcoul  = qq*rinv * (interact*rinvsq - fexcl);
                 vcoul  = qq*(interact*(rinv - sh_ewald_)
                                -(Vtab_(ri) - halfsp*frac*(Ftab_(ri) + fexcl)));
                 V.vc += vcoul;
#endif

#ifdef CALC_COULOMB
                 fscal = frLJ*rinvsq + fcoul;
#else
                 fscal = frLJ*rinvsq;
#endif
                 fx = fscal*dx;
                 fy = fscal*dy;
                 fz = fscal*dz;

                 V.fi[XX] += fx;
                 V.fi[YY] += fy;
                 V.fi[ZZ] += fz;

                 fj(k,XX) += fx;
                 fj(k,YY) += fy;
                 fj(k,ZZ) += fz;

             }, V_sum);

        Vvdw_ci += V_sum.vdw;
        Vc_ci   += V_sum.vc;

        fi(i,XX) += V_sum.fi[XX];
        fi(i,YY) += V_sum.fi[YY];
        fi(i,ZZ) += V_sum.fi[ZZ];
    }

    for (int i = 2; i < 4; i++)
    {
        int type_i_off  = typei(i);
        real q          = qi(i);

        struct v_energy V_sum;
        Kokkos::parallel_reduce
            (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k,struct v_energy& V)
             {
                 int type_j_off = typej(k);

                 // reading lj parameter causes indirect access
#ifndef HALF_LJ
                 real c6       = nbfp_(type_i_off + type_j_off);
                 real c12      = nbfp_(type_i_off + type_j_off + 1);
#endif
                 real dx, dy, dz;
                 real rsq, rinv;
                 real rinvsq, rinvsix;
                 real FrLJ6 = 0, FrLJ12 = 0, frLJ = 0, VLJ = 0;

#ifdef CALC_COULOMB
                 real qq;
                 real fcoul;
                 real rs, frac;
                 int  ri;
                 real fexcl;
                 real vcoul;
#endif
                 real fscal = 0.0;
                 real fx, fy, fz;

                 real skipmask;

#ifdef CHECK_EXCLS
                 /* A multiply mask used to zero an interaction
                  * when that interaction should be excluded
                  * (e.g. because of bonding). */
                 int interact;

                 interact = ((excl>>(i*UNROLLI + k)) & 1);

#ifndef EXCL_FORCES
                 skipmask = interact;
#else
                 skipmask = !(cj == ci_sh && k <= i);
#endif
#else
#define interact 1.0
                 skipmask = 1.0;
#endif
                 dx  = xi(i,XX) - xj(k,XX);
                 dy  = xi(i,YY) - xj(k,YY);
                 dz  = xi(i,ZZ) - xj(k,ZZ);

                 rsq = dx*dx + dy*dy + dz*dz;

                 // prevents vectorization (complains loop with early exit)
                 /* /\* Prepare to enforce the cut-off. *\/ */
                 skipmask = (rsq >= rcut2) ? 0 : skipmask;

#ifdef CHECK_EXCLS
                 /* Excluded atoms are allowed to be on top of each other.
                  * To avoid overflow of rinv, rinvsq and rinvsix
                  * we add a small number to rsq for excluded pairs only.
                  */
                 rsq += (1 - interact)*NBNXN_AVOID_SING_R2_INC;
#endif
                 rinv = 1.0/sqrt(rsq);

                 rinv = rinv * skipmask;

                 rinvsq  = rinv*rinv;

#ifndef HALF_LJ
                 rinvsix = interact*rinvsq*rinvsq*rinvsq;
                 FrLJ6   = c6*rinvsix;
                 FrLJ12  = c12*rinvsix*rinvsix;
                 frLJ    = FrLJ12 - FrLJ6;
                 VLJ     = (FrLJ12 + c12*repulsion_shift_cpot_)/12 -
                     (FrLJ6 + c6*dispersion_shift_cpot_)/6;

                 /* Need to zero the interaction if there should be exclusion. */
                 VLJ     = VLJ * interact;
                 /* Need to zero the interaction if r >= rcut */
                 VLJ     = VLJ * skipmask;

                 V.vdw +=  VLJ;
#endif

#ifdef CALC_COULOMB
                 qq     = skipmask * q * qj(k);
                 rs     = rsq*rinv*tabq_scale_;
                 ri     = (int)rs;
                 frac   = rs - ri;
                 fexcl  = (1 - frac)*Ftab_(ri) + frac*Ftab_(ri+1);
                 fcoul  = qq*rinv * (interact*rinvsq - fexcl);
                 vcoul  = qq*(interact*(rinv - sh_ewald_)
                                -(Vtab_(ri) - halfsp*frac*(Ftab_(ri) + fexcl)));

                 V.vc += vcoul;
#endif

#ifndef HALF_LJ
                 fscal = frLJ*rinvsq;
#endif

#ifdef CALC_COULOMB
                 fscal += fcoul;
#endif
                 fx = fscal*dx;
                 fy = fscal*dy;
                 fz = fscal*dz;

                 V.fi[XX] += fx;
                 V.fi[YY] += fy;
                 V.fi[ZZ] += fz;

                 fj(k,XX) += fx;
                 fj(k,YY) += fy;
                 fj(k,ZZ) += fz;

             }, V_sum);

        Vvdw_ci += V_sum.vdw;
        Vc_ci   += V_sum.vc;

        fi(i,XX) += V_sum.fi[XX];
        fi(i,YY) += V_sum.fi[YY];
        fi(i,ZZ) += V_sum.fi[ZZ];

    }

    Kokkos::parallel_for
        (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k)
         {
             int ajk = cj * UNROLLJ + k;
             f_view(ajk, XX) -= fj(k,XX);
             f_view(ajk, YY) -= fj(k,YY);
             f_view(ajk, ZZ) -= fj(k,ZZ);
         });


}

#undef interact
#undef EXCL_FORCES
