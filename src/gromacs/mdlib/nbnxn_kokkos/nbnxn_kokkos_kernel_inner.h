#if defined CHECK_EXCLS
#define EXCL_FORCES
#endif

{
    cj = cj_[I](cjind).cj;

    //wait until all threads load their xi and xj
    // dev.team_barrier();

    // load cluster j coordinates into shared memory
    Kokkos::parallel_for
        (Kokkos::ThreadVectorRange(dev,NBNXN_KOKKOS_CLUSTER_J_SIZE), [&] (const int& k)
         {
             int aj = cj * NBNXN_KOKKOS_CLUSTER_J_SIZE + k;
             xj_shmem[k*XI_STRIDE + XX] = x_(aj*XI_STRIDE + XX);
             xj_shmem[k*XI_STRIDE + YY] = x_(aj*XI_STRIDE + YY);
             xj_shmem[k*XI_STRIDE + ZZ] = x_(aj*XI_STRIDE + ZZ);
             qj_shmem[k] = facel_ * q_(aj);
             fj_sh.f[k*FI_STRIDE + XX] = 0.0;
             fj_sh.f[k*FI_STRIDE + YY] = 0.0;
             fj_sh.f[k*FI_STRIDE + ZZ] = 0.0;
         });
                    
    // compute forces on i atoms in parallel and reduce forces on j
    Kokkos::parallel_reduce 
        (Kokkos::ThreadVectorRange(dev,NBNXN_KOKKOS_CLUSTER_I_SIZE), [=] (const int& k, struct fj_reduce& fj)
         {
             for (int j = 0; j < NBNXN_KOKKOS_CLUSTER_J_SIZE; j++)
             {
                 int ai = ci * NBNXN_KOKKOS_CLUSTER_I_SIZE + k;
                 int i_type = ntype_*2*type_(ai);
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

                 interact = ((cj_[I](cjind).excl>>(k*NBNXN_KOKKOS_CLUSTER_I_SIZE + j)) & 1);
#ifndef EXCL_FORCES
                 skipmask = interact;
#else
                 skipmask = !(cj == ci_sh && j <= k);
#endif
#else
#define interact 1.0
                 skipmask = 1.0;
#endif

                 real dx = xi_shmem[k*XI_STRIDE + XX] - xj_shmem[j*XI_STRIDE + XX];
                 real dy = xi_shmem[k*XI_STRIDE + YY] - xj_shmem[j*XI_STRIDE + YY];
                 real dz = xi_shmem[k*XI_STRIDE + ZZ] - xj_shmem[j*XI_STRIDE + ZZ];
                 real rsq = dx*dx + dy*dy + dz*dz;

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

                 real rinv = gmx_invsqrt(rsq);
                 /* 5 flops for invsqrt */

                 /* Partially enforce the cut-off (and perhaps
                  * exclusions) to avoid possible overflow of
                  * rinvsix when computing LJ, and/or overflowing
                  * the Coulomb table during lookup. */
                 rinv = rinv * skipmask;

                 real rinvsq  = rinv*rinv;

                 // compute Coulomb force for Ewald type
                 real qq     = skipmask * qi_shmem[k] * qj_shmem[j];
                 real rs     = rsq*rinv*tabq_scale_;
                 int  ri     = (int)rs;
                 real frac   = rs - ri;
                 real fexcl  = (1 - frac)*Ftab_(ri) + frac*Ftab_(ri+1);

                 real fcoul  = qq*rinv*(interact*rinvsq - fexcl);

                 // compute LJ126 simple cut-off force
                 // \todo using a combination rule may be more memory efficient
                 int j_global = (cj * NBNXN_KOKKOS_CLUSTER_J_SIZE + j);
                 int j_type = type_(j_global);

                 real c6      = nbfp_(i_type + j_type*2 ); //0.26187E-02;   //nbfp_(i_type + j_type*2 );
                 real c12     = nbfp_(i_type + j_type*2+1 );//0.26307E-05;   //nbfp_(i_type + j_type*2+1 );

                 // printf("c6 = %lf",c6);
                 // printf("c12 = %lf",c12);

                 real rinvsix = interact*rinvsq*rinvsq*rinvsq;
                 real FrLJ6   = c6*rinvsix;
                 real FrLJ12  = c12*rinvsix*rinvsix;
                 real frLJ    = FrLJ12 - FrLJ6;

                 real fscal = frLJ*rinvsq + fcoul;

                 real fx = fscal*dx;
                 real fy = fscal*dy;
                 real fz = fscal*dz;

                 /* Increment i-atom force */
                 fi_shmem[k*FI_STRIDE + XX] += fx;
                 fi_shmem[k*FI_STRIDE + YY] += fy;
                 fi_shmem[k*FI_STRIDE + ZZ] += fz;

                 fj.f[j*FI_STRIDE + XX] -= fx;
                 fj.f[j*FI_STRIDE + YY] -= fy;
                 fj.f[j*FI_STRIDE + ZZ] -= fz;

             } // for loop over j
                            
         }, fj_sh); // parallel vector loop

                    // add j forces to global f
                    // atomic addition
    Kokkos::parallel_for
        (Kokkos::ThreadVectorRange(dev,NBNXN_KOKKOS_CLUSTER_J_SIZE), [&] (const int& k)
         {
             int aj = cj * NBNXN_KOKKOS_CLUSTER_I_SIZE + k;
             f_[I](aj*FI_STRIDE + XX) += fj_sh.f[k*FI_STRIDE + XX];
             f_[I](aj*FI_STRIDE + YY) += fj_sh.f[k*FI_STRIDE + YY];
             f_[I](aj*FI_STRIDE + ZZ) += fj_sh.f[k*FI_STRIDE + ZZ];
         });

}

#undef interact
#undef EXCL_FORCES
