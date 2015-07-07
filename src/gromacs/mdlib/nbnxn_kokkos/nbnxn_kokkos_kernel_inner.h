#if defined CHECK_EXCLS
#define EXCL_FORCES
#endif

{
    cj = cj_(cjind).cj;
    aj = cj * NBNXN_KOKKOS_CLUSTER_J_SIZE + it;

    fj_shmem[it*F_STRIDE + XX] = 0.0;
    fj_shmem[it*F_STRIDE + YY] = 0.0;
    fj_shmem[it*F_STRIDE + ZZ] = 0.0;
                
    xj_shmem[it*X_STRIDE + XX] = x_(aj*X_STRIDE + XX);
    xj_shmem[it*X_STRIDE + YY] = x_(aj*X_STRIDE + YY);
    xj_shmem[it*X_STRIDE + ZZ] = x_(aj*X_STRIDE + ZZ);

    qj_shmem[it] = q_(aj);

    //wait until all threads load their xi and xj
    dev.team_barrier();

    // each thread computes forces on its own i atom due to all j atoms in cj cluster
    // \todo this can be done using SIMD unit of each thread
    for (j = 0; j < NBNXN_KOKKOS_CLUSTER_J_SIZE; j++)
    {

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

        interact = ((cj_(cjind).excl>>(it*NBNXN_KOKKOS_CLUSTER_I_SIZE + j)) & 1);
#ifndef EXCL_FORCES
        skipmask = interact;
#else
        skipmask = !(cj == ci_sh && j <= it);
#endif
#else
#define interact 1.0
        skipmask = 1.0;
#endif

        dx = xi_shmem[it*X_STRIDE + XX] - xj_shmem[j*X_STRIDE + XX];
        dy = xi_shmem[it*X_STRIDE + YY] - xj_shmem[j*X_STRIDE + YY];
        dz = xi_shmem[it*X_STRIDE + ZZ] - xj_shmem[j*X_STRIDE + ZZ];
        rsq = dx*dx + dy*dy + dz*dz;

        /* Prepare to enforce the cut-off. */
        skipmask = (rsq >= rcut2_) ? 0 : skipmask;
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

        // compute Coulomb force for Ewald type
        qq     = skipmask * qi_shmem[it] * qj_shmem[j];
        rs     = rsq*rinv*tabq_scale_;
        ri     = (int)rs;
        frac   = rs - ri;
        fexcl  = (1 - frac)*Ftab_(ri) + frac*Ftab_(ri+1);

        fcoul  = qq*rinv*(interact*rinvsq - fexcl);

        // compute LJ126 simple cut-off force
        // \todo using a combination rule may be more memory efficient
        j_global = (cj * NBNXN_KOKKOS_CLUSTER_J_SIZE + j);
        j_type = type_(j_global);

        c6      = nbfp_(i_type + j_type*2 ); //0.26187E-02;   //nbfp_(i_type + j_type*2 );
        c12     = nbfp_(i_type + j_type*2+1 );//0.26307E-05;   //nbfp_(i_type + j_type*2+1 );

        // printf("c6 = %lf",c6);
        // printf("c12 = %lf",c12);

        rinvsix = interact*rinvsq*rinvsq*rinvsq;
        FrLJ6   = c6*rinvsix;
        FrLJ12  = c12*rinvsix*rinvsix;
        frLJ    = FrLJ12 - FrLJ6;

        fscal = frLJ*rinvsq + fcoul;

        fx = fscal*dx;
        fy = fscal*dy;
        fz = fscal*dz;


        // debug: checking forces on 1000th atoms

        // if (ai == 1000 && cj == 331)
        // {
        //     printf("cjind0 = %d cjind1 = %d \n", cjind0, cjind1);
        //     dy = xi_shmem[it*X_STRIDE + YY] - xj_shmem[j*X_STRIDE + YY];
        //     dz = xi_shmem[it*X_STRIDE + ZZ] - xj_shmem[j*X_STRIDE + ZZ];
        //     printf("ci = %d cj = %d j = %d \n", ci, cj, j);
        //     printf("xi = %lf %lf %lf \n",xi_shmem[it*X_STRIDE + XX],xi_shmem[it*X_STRIDE + YY],xi_shmem[it*X_STRIDE + ZZ]);
        //     printf("xj = %lf %lf %lf \n",xj_shmem[j*X_STRIDE + XX],xj_shmem[j*X_STRIDE + YY],xj_shmem[j*X_STRIDE + ZZ]);
        //     printf("i = %d dx = %lf fx = %lf \n", ai, dx, fx);
        //     printf("i = %d dy = %lf fy = %lf \n", ai, dy, fy);
        //     printf("i = %d dz = %lf fz = %lf \n", ai, dz, fz);
        // }   
        /* Increment i-atom force */
        fi_shmem[it*F_STRIDE + XX] += fx;
        fi_shmem[it*F_STRIDE + YY] += fy;
        fi_shmem[it*F_STRIDE + ZZ] += fz;

        // /* Decrement j-atom force */
        // do atomically: since all the threads write to fj_shmem
        Kokkos::atomic_add(&fj_shmem[j*F_STRIDE + XX], -1.0*fx);
        Kokkos::atomic_add(&fj_shmem[j*F_STRIDE + YY], -1.0*fy);
        Kokkos::atomic_add(&fj_shmem[j*F_STRIDE + ZZ], -1.0*fz);

        // total forces on current j atom due to all i atoms in a thread team
        // fx_total = dev.reduce(fx);
        // fy_total = dev.reduce(fy);
        // fz_total = dev.reduce(fz);

        // // Add with one thread and vectorlane of the team
        // Kokkos::single(Kokkos::PerTeam(dev),[=] () {
        //         // atomicaly add value to global f
        //         f_(j_global*F_STRIDE + XX) -= fx_total;
        //         f_(j_global*F_STRIDE + YY) -= fy_total;
        //         f_(j_global*F_STRIDE + ZZ) -= fz_total;
        //     });

    } // loop over j atoms

    //wait until all threads computed their forces btw i and j in current cj
    dev.team_barrier();
    f_(aj*F_STRIDE + XX) += fj_shmem[it*F_STRIDE + XX];
    f_(aj*F_STRIDE + YY) += fj_shmem[it*F_STRIDE + YY];
    f_(aj*F_STRIDE + ZZ) += fj_shmem[it*F_STRIDE + ZZ];
    dev.team_barrier();

}

#undef interact
#undef EXCL_FORCES
