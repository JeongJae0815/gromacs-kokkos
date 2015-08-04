#if defined CHECK_EXCLS && (defined CALC_COULOMB || defined LJ_EWALD)
#define EXCL_FORCES
#endif

{
    const int cj = cj_[I](cjind).cj;
    const unsigned int excl = cj_[I](cjind).excl;

    int cjj         = cj*UNROLLJ;
    /* Kokkos::parallel_for */
    /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k) */
    for(int k = 0; k < UNROLLJ; k++)
         {
             int ajk = cjj + k;

             xj(k,XX) = x_(ajk,XX);
             xj(k,YY) = x_(ajk,YY);
             xj(k,ZZ) = x_(ajk,ZZ);

             qj(k)    =  q_(ajk);

             fj(k,XX) = 0.0;
             fj(k,YY) = 0.0;
             fj(k,ZZ) = 0.0;

         }//);

    // DOES NOT VECTORIZE
    Kokkos::parallel_for
        (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k)
         {
             int ajk = cjj + k;
             typej(k) = type_(ajk)*2;
         });

    for (int i = 0; i < 4; i++)
    {
        int type_i_off  = typei(i);
        real q          = qi(i);

        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 dx(k,ZZ)  = xi(i,ZZ) - xj(k,ZZ);
                 dx(k,YY)  = xi(i,YY) - xj(k,YY);
                 dx(k,XX)  = xi(i,XX) - xj(k,XX);
             }//);

        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 rsq(k)  = dx(k,XX)*dx(k,XX) + dx(k,YY)*dx(k,YY) + dx(k,ZZ)*dx(k,ZZ);
             }//);
        
        real skipmask[UNROLLJ] = {1.0,1.0,1.0,1.0};
        int interact[UNROLLJ] = {1,1,1,1};

#ifdef CHECK_EXCLS

        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 interact[k] = ((excl>>(i*UNROLLI + k)) & 1);
             }//);

#ifndef EXCL_FORCES

        Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k)
        /* for(int k = 0; k < UNROLLJ; k++) */
             {
                 skipmask[k] = interact[k];
             });

#else

        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 skipmask[k] = !(cj == ci_sh && k <= i);
             }//);

#endif // EXCL_FORCES
#endif // CHECK_EXCLS

        // DOESN'T Vectorize due to if condition
        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 skipmask[k] = (rsq[k] >= rcut2) ? 0 : skipmask[k];
             }//);

#ifdef CHECK_EXCLS

        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 rsq(k)  += (1 - interact[k])*NBNXN_AVOID_SING_R2_INC;
             }//);

#endif

        // ThIS ONE VECTORIZES AND IMPROVES PERFORMANCE
        Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k)
        /* for(int k = 0; k < UNROLLJ; k++) */
             {
                 rinv(k)  = skipmask[k]*1.0/sqrt(rsq(k));
             });

        real rinvsq[UNROLLJ];
        real rinvsix[UNROLLJ];

        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 rinvsq[k] = rinv(k)*rinv(k);
             }//);

        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 rinvsix[k] = interact[k]*rinvsq[k]*rinvsq[k]*rinvsq[k];
             }//);

        real c6[UNROLLJ];
        real c12[UNROLLJ];
        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 c6[k]  = nbfp_(type_i_off + typej(k));
                 c12[k] = nbfp_(type_i_off + typej(k) + 1);
             }//);

        real FrLJ12[UNROLLJ];
        real FrLJ6[UNROLLJ];
        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 FrLJ6[k]   = c6[k]*rinvsix[k];
                 FrLJ12[k]  = c12[k]*rinvsix[k]*rinvsix[k];
             }//);

        real frLJ[UNROLLJ];
        real VLJ[UNROLLJ];
        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 frLJ[k]    = FrLJ12[k] - FrLJ6[k];
                 VLJ[k]     = interact[k] * skipmask[k] * ( (FrLJ12[k] + c12[k]*repulsion_shift_cpot_)/12 -
                                                            (FrLJ6[k] + c6[k]*dispersion_shift_cpot_)/6 );
             }//);

#ifdef CALC_COULOMB
        real qq[UNROLLJ];
        real rs[UNROLLJ];
        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 qq[k] = q * qj(k);
                 rs[k] = rsq(k) * rinv(k) * tabq_scale_;
             }//);

        int ri[UNROLLJ];
        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 ri[k] = (int)rs[k];
             }//);

        real frac[UNROLLJ];
        Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k)
        /* for(int k = 0; k < UNROLLJ; k++) */
             {
                 frac[k] = rs[k] - ri[k];
             });

        real fexcl[UNROLLJ];
        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 fexcl[k] = (1 - frac[k])*Ftab_(ri[k]) + frac[k]*Ftab_(ri[k]+1);
             }//);

        real fcoul[UNROLLJ];
        real vcoul[UNROLLJ];
        Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k)
        /* for(int k = 0; k < UNROLLJ; k++) */
             {
                 fcoul[k] = qq[k]*rinv(k) * (interact[k]*rinvsq[k] - fexcl[k]);
                 vcoul[k]  = qq[k]*(interact[k]*(rinv(k) - sh_ewald_)
                                    -(Vtab_(ri[k]) - halfsp*frac[k]*(Ftab_(ri[k]) + fexcl[k])));
             });

#endif

        real fscal[UNROLLJ];
#ifdef CALC_COULOMB

        Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k)
        /* for(int k = 0; k < UNROLLJ; k++) */
             {
                 fscal[k] = frLJ[k]*rinvsq[k] + fcoul[k];
             });

#else

        Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(dev,UNROLLJ), [&] (int& k)
        /* for(int k = 0; k < UNROLLJ; k++) */
             {
                 fscal[k] = frLJ[k]*rinvsq[k];
             });

#endif

        /* Kokkos::parallel_for */
        /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k) */
        for(int k = 0; k < UNROLLJ; k++)
             {
                 fi(i,XX) += fscal[k]*dx(k,XX);
                 fi(i,YY) += fscal[k]*dx(k,YY);
                 fi(i,ZZ) += fscal[k]*dx(k,ZZ);

                 fj(k,XX) -= fscal[k]*dx(k,XX);
                 fj(k,YY) -= fscal[k]*dx(k,YY);
                 fj(k,ZZ) -= fscal[k]*dx(k,ZZ);
             }//);

        real vvdw = 0.0;
        Kokkos::parallel_reduce
            (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k,real& v)
             {
                 v += VLJ[k];
             },vvdw);

        Vvdw_ci += vvdw;

#ifdef CALC_COULOMB
        real vc = 0.0;
        Kokkos::parallel_reduce
            (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k,real& v)
             {
                 v += vcoul[k];
             },vc);
        Vc_ci += vc;
#endif

    }

    /* Kokkos::parallel_for */
    /*     (Kokkos::ThreadVectorRange(dev,UNROLLJ), KOKKOS_LAMBDA (int& k) */
    for(int k = 0; k < UNROLLJ; k++)
         {
             int ajk = cj * UNROLLJ + k;
             f_view(ajk, XX) += fj(k,XX);
             f_view(ajk, YY) += fj(k,YY);
             f_view(ajk, ZZ) += fj(k,ZZ);
         }//);
        
}

#undef interact
#undef EXCL_FORCES
