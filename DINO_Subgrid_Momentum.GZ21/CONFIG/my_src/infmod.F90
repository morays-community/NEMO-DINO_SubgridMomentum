MODULE infmod
   !!======================================================================
   !!                       ***  MODULE  infmod  ***
   !! Machine Learning Inferences : manage connexion with external ML codes 
   !!======================================================================
   !! History :  4.2.1  ! 2023-09  (A. Barge)  Original code
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   naminf          : machine learning models formulation namelist
   !!   inferences_init : initialization of Machine Learning based models
   !!   inferences      : ML based models
   !!   inf_snd         : send data to external trained model
   !!   inf_rcv         : receive inferences from external trained model
   !!----------------------------------------------------------------------
   USE oce             ! ocean fields
   USE dom_oce         ! ocean domain fields
   USE inffld          ! working fields for inferences models
   USE cpl_oasis3      ! OASIS3 coupling
   USE timing
   USE iom
   USE in_out_manager
   USE lib_mpp

   IMPLICIT NONE
   PRIVATE

   PUBLIC inf_alloc          ! function called in inferences_init 
   PUBLIC inf_dealloc        ! function called in inferences_final
   PUBLIC inferences_init    ! routine called in nemogcm.F90
   PUBLIC inferences_send    ! routine called in stpmlf.F90
   PUBLIC inferences_rcv     ! routine called in stpmlf.F90
   PUBLIC inferences_final   ! routine called in nemogcm.F90

   INTEGER, PARAMETER ::   jps_ssu = 1    ! u surface velocity
   INTEGER, PARAMETER ::   jps_ssv = 2    ! v surface velocity
   INTEGER, PARAMETER ::   jps_mu = 3    ! u mask
   INTEGER, PARAMETER ::   jps_mv = 4    ! u mask
   INTEGER, PARAMETER ::   jps_inf = 4   ! total number of sendings for inferences

   INTEGER, PARAMETER ::   jpr_uf = 1    ! u subgrid forcing field
   INTEGER, PARAMETER ::   jpr_vf = 2    ! v subgrid forcing field
   INTEGER, PARAMETER ::   jpr_inf = 2   ! total number of inference receptions

   INTEGER, PARAMETER ::   jpinf = MAX(jps_inf,jpr_inf) ! Maximum number of exchanges

   TYPE( DYNARR ), SAVE, DIMENSION(jpinf) ::  infsnd, infrcv  ! sent/received inferences

   !
   !!-------------------------------------------------------------------------
   !!                    Namelist for the Inference Models
   !!-------------------------------------------------------------------------
   !                           !!** naminf namelist **
   !TYPE ::   FLD_INF              !: Field informations ...  
   !   CHARACTER(len = 32) ::         ! 
   !END TYPE FLD_INF
   !
   LOGICAL , PUBLIC ::   ln_inf        !: activate module for inference models
   INTEGER , PUBLIC ::   nn_lvl        !: number of grid level on which apply the inference model
   
   !!-------------------------------------------------------------------------

CONTAINS

   INTEGER FUNCTION inf_alloc()
      !!----------------------------------------------------------------------
      !!             ***  FUNCTION inf_alloc  ***
      !!----------------------------------------------------------------------
      INTEGER :: ierr
      INTEGER :: jn
      !!----------------------------------------------------------------------
      ierr = 0
      !
      DO jn = 1, jpinf
         IF( srcv(ntypinf,jn)%laction ) ALLOCATE( infrcv(jn)%z3(jpi,jpj,srcv(ntypinf,jn)%nlvl), STAT=ierr )
         IF( ssnd(ntypinf,jn)%laction ) ALLOCATE( infsnd(jn)%z3(jpi,jpj,ssnd(ntypinf,jn)%nlvl), STAT=ierr )
         inf_alloc = MAX(ierr,0)
      END DO
      !
   END FUNCTION inf_alloc

   
   INTEGER FUNCTION inf_dealloc()
      !!----------------------------------------------------------------------
      !!             ***  FUNCTION inf_dealloc  ***
      !!----------------------------------------------------------------------
      INTEGER :: ierr
      INTEGER :: jn
      !!----------------------------------------------------------------------
      ierr = 0
      !
      DO jn = 1, jpinf
         IF( srcv(ntypinf,jn)%laction ) DEALLOCATE( infrcv(jn)%z3, STAT=ierr )
         IF( ssnd(ntypinf,jn)%laction ) DEALLOCATE( infsnd(jn)%z3, STAT=ierr )
         inf_dealloc = MAX(ierr,0)
      END DO
      !
   END FUNCTION inf_dealloc


   SUBROUTINE inferences_init 
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE inferences_init  ***
      !!
      !! ** Purpose :   Initialisation of the models that rely on external inferences
      !!
      !! ** Method  :   * Read naminf namelist
      !!                * create data for models
      !!----------------------------------------------------------------------
      !
      INTEGER ::   ios   ! Local Integer
      !!
      NAMELIST/naminf/  ln_inf, nn_lvl
      !!----------------------------------------------------------------------
      !
      ! ================================ !
      !      Namelist informations       !
      ! ================================ !
      !
      READ  ( numnam_ref, naminf, IOSTAT = ios, ERR = 901)
901   IF( ios /= 0 )   CALL ctl_nam ( ios , 'naminf in reference namelist' )
      !
      READ  ( numnam_cfg, naminf, IOSTAT = ios, ERR = 902 )
902   IF( ios >  0 )   CALL ctl_nam ( ios , 'naminf in configuration namelist' )
      IF( lwm ) WRITE ( numond, naminf )
      !
      IF( lwp ) THEN                        ! control print
         WRITE(numout,*)
         WRITE(numout,*)'inferences_init : Setting inferences models'
         WRITE(numout,*)'~~~~~~~~~~~~~~~'
      END IF
      IF ( lwp .AND. ln_inf ) THEN
         WRITE(numout,*)'   Namelist naminf'
         WRITE(numout,*)'      Module used       ln_inf        = ', ln_inf
         WRITE(numout,*)'      Model applied on nn_lvl grid level,   nn_lvl     = ', nn_lvl
      ENDIF
      !
      IF( ln_inf .AND. .NOT. lk_oasis )   CALL ctl_stop( 'inferences_init : External inferences coupled via OASIS, but key_oasis3 disabled' )
      !
      !
      ! ======================================== !
      !     Define exchange needs for Models     !
      ! ======================================== !
      !
      ! default definitions of ssnd snd srcv
      srcv(ntypinf,:)%laction = .FALSE.  ;  srcv(ntypinf,:)%clgrid = 'T'  ;  srcv(ntypinf,:)%nsgn = 1.
      srcv(ntypinf,:)%nct = 1  ;  srcv(ntypinf,:)%nlvl = nn_lvl
      !
      ssnd(ntypinf,:)%laction = .FALSE.  ;  ssnd(ntypinf,:)%clgrid = 'T'  ;  ssnd(ntypinf,:)%nsgn = 1.
      ssnd(ntypinf,:)%nct = 1  ;  ssnd(ntypinf,:)%nlvl = nn_lvl
      
      IF( ln_inf ) THEN
      
         ! -------------------------------- !
         !      Kenigson et al. (2022)      !
         ! -------------------------------- !

         ! sending of sea surface U velocity
         ssnd(ntypinf,jps_ssu)%clname = 'E_OUT_0'
         ssnd(ntypinf,jps_ssu)%laction = .TRUE.
         ssnd(ntypinf,jps_ssu)%clgrid = 'U'

         ! sending of sea surface V velocity
         ssnd(ntypinf,jps_ssv)%clname = 'E_OUT_1'
         ssnd(ntypinf,jps_ssv)%laction = .TRUE.
         ssnd(ntypinf,jps_ssv)%clgrid = 'V'

         ! sending of u-grid and v-grid masks
         ssnd(ntypinf,jps_mu)%clname = 'E_OUT_2'
         ssnd(ntypinf,jps_mu)%laction = .TRUE.
         ssnd(ntypinf,jps_mu)%clgrid = 'U'

         ssnd(ntypinf,jps_mv)%clname = 'E_OUT_3'
         ssnd(ntypinf,jps_mv)%laction = .TRUE.
         ssnd(ntypinf,jps_mv)%clgrid = 'V'

         ! reception of u-grid and v-grid subgrid forcing fields
         srcv(ntypinf,jpr_uf)%clname = 'E_IN_0'
         srcv(ntypinf,jpr_uf)%laction = .TRUE.
         srcv(ntypinf,jpr_uf)%clgrid = 'U'

         srcv(ntypinf,jpr_vf)%clname = 'E_IN_1'
         srcv(ntypinf,jpr_vf)%laction = .TRUE.
         srcv(ntypinf,jpr_vf)%clgrid = 'V'

         ! ------------------------------ !
         ! ------------------------------ !

      END IF
      ! 
      ! ================================= !
      !   Define variables for coupling
      ! ================================= !
      CALL cpl_var(jpinf, jpinf, 1, ntypinf)
      !
      IF( inf_alloc() /= 0 )     CALL ctl_stop( 'STOP', 'inf_alloc : unable to allocate arrays' )
      IF( inffld_alloc() /= 0 )  CALL ctl_stop( 'STOP', 'inffld_alloc : unable to allocate arrays' ) 
      !
   END SUBROUTINE inferences_init


   SUBROUTINE inferences_send( kt, Kbb, puu, pvv )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE inferences  ***
      !!
      !! ** Purpose :   send surface velocity fields toward Python script
      !!
      !! ** Method  :   *  
      !!                * 
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in)                                 ::  kt             ! ocean time step
      INTEGER, INTENT(in)                                 ::  Kbb            ! ocean time level indices
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(inout) ::  puu, pvv       ! ocean horizontal velocities
      !
      INTEGER :: isec, info, jn                       ! local integer
      REAL(wp), DIMENSION(jpi,jpj,jpk)   ::  zdata    ! sending buffer
      !!----------------------------------------------------------------------
      !
      isec = ( kt - nit000 ) * NINT( rn_Dt )       ! Date of exchange 
      info = OASIS_idle
      !
      ! ------  Prepare data to send ------
      !
      ! Sea Surface U velocity
      IF( ssnd(ntypinf,jps_ssu)%laction ) THEN
         infsnd(jps_ssu)%z3(:,:,1:ssnd(ntypinf,jps_ssu)%nlvl) = puu(:,:,1:ssnd(ntypinf,jps_ssu)%nlvl,Kbb)
      ENDIF  
      !
      ! Sea Surface V velocity
      IF( ssnd(ntypinf,jps_ssv)%laction ) THEN
         infsnd(jps_ssv)%z3(:,:,1:ssnd(ntypinf,jps_ssv)%nlvl) = pvv(:,:,1:ssnd(ntypinf,jps_ssv)%nlvl,Kbb) 
      ENDIF
      !
      ! u-grid surface mask
      IF( ssnd(ntypinf,jps_mu)%laction ) THEN
          infsnd(jps_mu)%z3(:,:,1:ssnd(ntypinf,jps_mu)%nlvl) = umask(:,:,1:ssnd(ntypinf,jps_mu)%nlvl)
      ENDIF
      !
      ! v-grid surface mask
      IF( ssnd(ntypinf,jps_mv)%laction ) THEN
          infsnd(jps_mv)%z3(:,:,1:ssnd(ntypinf,jps_mv)%nlvl) = vmask(:,:,1:ssnd(ntypinf,jps_mv)%nlvl)
      ENDIF
      !
      ! ========================
      !   Proceed all sendings
      ! ========================
      !
      DO jn = 1, jpinf
         IF ( ssnd(ntypinf,jn)%laction ) THEN
            CALL cpl_snd( jn, isec, ntypinf, infsnd(jn)%z3, info)
         ENDIF
      END DO
      !
   END SUBROUTINE inferences_send


   SUBROUTINE inferences_rcv( kt, puu, pvv, Krhs )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE inferences  ***
      !!
      !! ** Purpose :   get forcing fields from Python script, add it to RHS
      !!
      !! ** Method  :   *
      !!                *
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in)                                 ::  kt             ! ocean time step
      INTEGER, INTENT(in)                                 ::  Krhs           ! ocean time level indices
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(inout) ::  puu, pvv       ! ocean horizontal velocities
      !
      INTEGER :: isec, info, jn                       ! local integer
      REAL(wp), DIMENSION(jpi,jpj,jpk)   ::  zdata    ! sending buffer
      !!----------------------------------------------------------------------
      !
      isec = ( kt - nit000 ) * NINT( rn_Dt )       ! Date of exchange
      info = OASIS_idle
      !
      IF( ln_timing )   CALL timing_start('inferences')
      !
      ! ==========================
      !   Proceed all receptions
      ! ==========================
      !
      DO jn = 1, jpinf
         IF( srcv(ntypinf,jn)%laction ) THEN
            CALL cpl_rcv( jn, isec, ntypinf, infrcv(jn)%z3, info)
         ENDIF
      END DO
      !
      ! ------ Distribute receptions  ------
      !
      ! Subgrid forcing fields
      IF( srcv(ntypinf,jpr_uf)%laction .AND. srcv(ntypinf,jpr_vf)%laction ) THEN
         ext_uf(:,:,1:srcv(ntypinf,jpr_uf)%nlvl) = infrcv(jpr_uf)%z3(:,:,1:srcv(ntypinf,jpr_uf)%nlvl)
         ext_vf(:,:,1:srcv(ntypinf,jpr_vf)%nlvl) = infrcv(jpr_vf)%z3(:,:,1:srcv(ntypinf,jpr_vf)%nlvl)

         ! Output surface tendencies 
         CALL iom_put( 'ext_uf', ext_uf(:,:,1) )
         CALL iom_put( 'ext_vf', ext_vf(:,:,1) )

         ! Apply forcing fields to RHS
         puu(:,:,1:srcv(ntypinf,jpr_uf)%nlvl,Krhs) = puu(:,:,1:srcv(ntypinf,jpr_uf)%nlvl,Krhs) + ext_uf(:,:,1:srcv(ntypinf,jpr_uf)%nlvl)
         pvv(:,:,1:srcv(ntypinf,jpr_vf)%nlvl,Krhs) = pvv(:,:,1:srcv(ntypinf,jpr_vf)%nlvl,Krhs) + ext_vf(:,:,1:srcv(ntypinf,jpr_vf)%nlvl)
      ENDIF
      !
      IF( ln_timing )   CALL timing_stop('inferences')
      !
   END SUBROUTINE inferences_rcv


   SUBROUTINE inferences_final
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE inferences_final  ***
      !!
      !! ** Purpose :   Free memory used for inferences modules
      !!
      !! ** Method  :   * Deallocate arrays
      !!----------------------------------------------------------------------
      !
      IF( inf_dealloc() /= 0 )     CALL ctl_stop( 'STOP', 'inf_dealloc : unable to free memory' )
      IF( inffld_dealloc() /= 0 )  CALL ctl_stop( 'STOP', 'inffld_dealloc : unable to free memory' )      
      !
   END SUBROUTINE inferences_final 
   !!=======================================================================
END MODULE infmod
