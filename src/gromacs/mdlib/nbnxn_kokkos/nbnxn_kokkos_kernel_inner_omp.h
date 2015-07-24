#if defined CHECK_EXCLS && (defined CALC_COULOMB || defined LJ_EWALD)
#define EXCL_FORCES
#endif

{
    int cj;
    int i, j;

    cj = cj_[I](cjind).cj;
    const unsigned int excl = cj_[I](cjind).excl;

    Kokkos::parallel_for
        (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (const int& k)
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

    for (i = 0; i < UNROLLI; i++)
    {
        int ai;
        int type_i_off;

        ai = ci*UNROLLI + i;

        type_i_off = type_(ai)*ntype2;

        for (j = 0; j < UNROLLJ; j++)
        {
            int  aj;
            real dx, dy, dz;
            real rsq, rinv;
            real rinvsq, rinvsix;
            real c6, c12;
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

            /* A multiply mask used to zero an interaction
             * when either the distance cutoff is exceeded, or
             * (if appropriate) the i and j indices are
             * unsuitable for this kind of inner loop. */
            real skipmask;

#ifdef CHECK_EXCLS
            /* A multiply mask used to zero an interaction
             * when that interaction should be excluded
             * (e.g. because of bonding). */
            int interact;

            interact = ((excl>>(i*UNROLLI + j)) & 1);
#ifndef EXCL_FORCES
            skipmask = interact;
#else
            skipmask = !(cj == ci_sh && j <= i);
#endif
#else
#define interact 1.0
            skipmask = 1.0;
#endif

            aj = cj*UNROLLJ + j;

            dx  = xi(i,XX) - xj(j,XX);
            dy  = xi(i,YY) - xj(j,YY);
            dz  = xi(i,ZZ) - xj(j,ZZ);

            rsq = dx*dx + dy*dy + dz*dz;

            /* Prepare to enforce the cut-off. */
            skipmask = (rsq >= rcut2) ? 0 : skipmask;
            /* 9 flops for r^2 + cut-off check */

#ifdef CHECK_EXCLS
            /* Excluded atoms are allowed to be on top of each other.
             * To avoid overflow of rinv, rinvsq and rinvsix
             * we add a small number to rsq for excluded pairs only.
             */
            rsq += (1 - interact)*NBNXN_AVOID_SING_R2_INC;
#endif

            rinv = gmx_invsqrt(rsq);
            /* 5 flops for invsqrt */

            /* Partially enforce the cut-off (and perhaps
             * exclusions) to avoid possible overflow of
             * rinvsix when computing LJ, and/or overflowing
             * the Coulomb table during lookup. */
            rinv = rinv * skipmask;

            rinvsq  = rinv*rinv;

#ifdef HALF_LJ
            if (i < UNROLLI/2)
#endif
            {
                c6      = nbfp_(type_i_off+type_(aj)*2);
                c12     = nbfp_(type_i_off+type_(aj)*2+1);

                rinvsix = interact*rinvsq*rinvsq*rinvsq;
                FrLJ6   = c6*rinvsix;
                FrLJ12  = c12*rinvsix*rinvsix;
                frLJ    = FrLJ12 - FrLJ6;
                /* 7 flops for r^-2 + LJ force */
                VLJ     = (FrLJ12 + c12*repulsion_shift_cpot_)/12 -
                    (FrLJ6 + c6*dispersion_shift_cpot_)/6;

                /* Masking should be done after force switching,
                 * but before potential switching.
                 */
                /* Need to zero the interaction if there should be exclusion. */
                VLJ     = VLJ * interact;
                /* Need to zero the interaction if r >= rcut */
                VLJ     = VLJ * skipmask;
                /* 1 more flop for LJ energy */

                Vvdw_ci += VLJ;

            }

#ifdef CALC_COULOMB
            /* Enforce the cut-off and perhaps exclusions. In
             * those cases, rinv is zero because of skipmask,
             * but fcoul and vcoul will later be non-zero (in
             * both RF and table cases) because of the
             * contributions that do not depend on rinv. These
             * contributions cannot be allowed to accumulate
             * to the force and potential, and the easiest way
             * to do this is to zero the charges in
             * advance. */
            qq = skipmask * qi(i) * qj(j);

            rs     = rsq*rinv*tabq_scale_;
            ri     = (int)rs;
            frac   = rs - ri;
            /* fexcl = (1-frac) * F_i + frac * F_(i+1) */
            fexcl  = (1 - frac)*Ftab_(ri) + frac*Ftab_(ri+1);
            fcoul  = interact*rinvsq - fexcl;
            /* 7 flops for float 1/r-table force */

            vcoul  = qq*(interact*(rinv - sh_ewald_)
                         -(Vtab_(ri)
                           -halfsp*frac*(Ftab_(ri) + fexcl)));

            fcoul *= qq*rinv;

            Vc_ci += vcoul;

#endif

#ifdef CALC_COULOMB
#ifdef HALF_LJ
            if (i < UNROLLI/2)
#endif
            {
                fscal = frLJ*rinvsq + fcoul;
                /* 2 flops for scalar LJ+Coulomb force */
            }
#ifdef HALF_LJ
            else
            {
                fscal = fcoul;
            }
#endif
#else
            fscal = frLJ*rinvsq;
#endif
            fx = fscal*dx;
            fy = fscal*dy;
            fz = fscal*dz;

            /* Increment i-atom force */
            fi(i,XX) += fx;
            fi(i,YY) += fy;
            fi(i,ZZ) += fz;

            /* Decrement j-atom force */
            fj(j,XX) -= fx;
            fj(j,YY) -= fy;
            fj(j,ZZ) -= fz;
            /* 9 flops for force addition */
        }
    }

    Kokkos::parallel_for
        (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (const int& k)
         {
             int ajk = cj * UNROLLJ + k;

             f_[I](ajk, XX) += fj(k,XX);
             f_[I](ajk, YY) += fj(k,YY);
             f_[I](ajk, ZZ) += fj(k,ZZ);
         });

}

#undef interact
#undef EXCL_FORCES
