library( subtyper )
library( ANTsR )
library( rsq )
ukbfn = "data/allcsvs/ukb672504_imaging_long.csv"
if ( ! exists("dd") ) {
    dd=read.csv(ukbfn)
    dd=dd[ fs( dd$T1Hier_resnetGrade > 1 ), ]
    dd=dd[ !is.na(dd$DTI_dti_FD_mean) & !is.na( dd$rsfMRI_rsf_FD_mean ),  ]
    dd$subjectAge_BL=dd$age_MRI
    usubs = table( dd[,'eid']  )
    longsubs=usubs[usubs>1]
    for ( u in names(longsubs) ) {
        usel=dd[,'eid'] == u
        dd$subjectAge_BL[usel]=min(dd$age_MRI[usel])
        }
    dd$Years.bl=dd$age_MRI-dd$subjectAge_BL
    dd$DX='CN'
    subct=table( dd$eid )
    crosssubs=names(subct[subct==1])
    longsubs=names(subct[subct==2])
    print(paste("Cross:",length(crosssubs),"long:",length(longsubs)))
    t1names=c( 
        getNamesFromDataframe( "T1Hier_vol", dd, 
            exclusions=c("_BL","_delta","evratio","Grade","mhdist","RandB","outli","templateL1") ),
        getNamesFromDataframe( "T1Hier_thk", dd, 
            exclusions=c("_BL","_delta","evratio","Grade","mhdist","RandB","outli","templateL1") ) )
    dtinames=getNamesFromDataframe( "DTI"  , dd, exclusions=c("_BL","_delta","cnx","FD","motion","SNR","evr","ssnr","dvars","tsnr","volume") )
    rsfnames=getNamesFromDataframe( "rsfMRI"  , dd, exclusions=c("_BL","_delta","cnx","FD","motion","SNR","evr","ssnr","dvars","tsnr","volume","_alff") )

    #### cross-sectional
    dd$subjectAge_BL=antsrimpute(dd$subjectAge_BL)
    rsfdd = antsrimpute(scale( dd[,rsfnames] ))
    dtidd = antsrimpute(scale( dd[,dtinames] ))
    t1dd = antsrimpute(scale( dd[,t1names] ))
    t1dd = residuals( lm( t1dd ~ dd$sex_f31_0_0 + dd$subjectAge_BL))
    dtidd = residuals( lm( dtidd ~ dd$sex_f31_0_0 + dd$subjectAge_BL))
    rsfdd = residuals( lm( rsfdd ~ dd$sex_f31_0_0 + dd$subjectAge_BL))
    xsel = dd$eid %in% crosssubs
    write.csv( rsfdd[xsel,], "/tmp/ukrsfx.csv", row.names=FALSE)
    write.csv( dtidd[xsel,], "/tmp/ukdtix.csv", row.names=FALSE)
    write.csv( t1dd[xsel,], "/tmp/ukt1x.csv", row.names=FALSE)
    #### longitudinal
    lsel = dd$eid %in% longsubs
    write.csv( rsfdd[lsel,], "/tmp/ukrsfl.csv", row.names=FALSE)
    write.csv( dtidd[lsel,], "/tmp/ukdtil.csv", row.names=FALSE)
    write.csv( t1dd[lsel,], "/tmp/ukt1l.csv", row.names=FALSE)
    }

ev0=read.csv("/tmp/ukt1ev.csv")[,-1]
ev1=read.csv("/tmp/ukdtiev.csv")[,-1]
ev2=read.csv("/tmp/ukrsfev.csv")[,-1]

ev0=read.csv("/tmp/ukt1ev_b.csv")[,-1]
ev1=read.csv("/tmp/ukdtiev_b.csv")[,-1]
ev2=read.csv("/tmp/ukrsfev_b.csv")[,-1]

#
pt1=data.frame(data.matrix(t1dd[,]) %*% t(data.matrix(ev0)))
colnames(pt1)=paste0("simlrt1",1:ncol(pt1))
pdti=data.frame(data.matrix(dtidd[,]) %*% t(data.matrix(ev1)))
colnames(pdti)=paste0("simlrdti",1:ncol(pt1))
prsf=data.frame(data.matrix(rsfdd[,]) %*% t(data.matrix(ev2)))
colnames(prsf)=paste0("simlrrsf",1:ncol(pt1))
extras=cbind(pt1[,],pdti[,],prsf[,])
library(lmerTest)
ee=cbind(dd[,],extras)
nsim=nsimu=ncol(pt1)
gpca=paste(getNamesFromDataframe("genetic_principal_components",dd)[1:10]
,collapse="+")
simnames=paste(
    paste0(paste0("simlrt1",1:nsim),collapse='+'), "+",
    paste0(paste0("simlrdti",1:nsim),collapse='+'),"+",
    paste0(paste0("simlrrsf",1:nsim),collapse='+')
)
mf=paste("age_MRI~(1|eid)+sex_f31_0_0+",simnames,"+",gpca)
print(summary(lmer(mf,data=ee)))


intnames=getNamesFromDataframe("luid",dd)
for ( i in intnames) {
    print(i)
    print( table( !is.na( ee[,i]) ))
}

for ( p in c("townsend_deprivation_index_at_recruitment_f22189_0_0","fluid_intelligence_score_f20016")) {
    mf0=paste(p,"~subjectAge_BL+sex_f31_0_0+",gpca)
    mf=paste(p,"~subjectAge_BL+sex_f31_0_0+",simnames,"+",gpca)
    bmdl=lm(mf0,data=ee)
    mdl=lm(mf,data=ee)
    print(summary(mdl))
    print( paste( p, rsq(bmdl), rsq(mdl) ))
}


# some clustering
#
nclust = 3
ctype='angle'
print(paste("BEGIN",ctype," learning CLUSTER-ing {{o/o}} k = ",nclust))
myclust = trainSubtypeClusterMulti(
       ee[xsel,],
       measureColumns=colnames(extras),
       method = ctype,
       nclust  )
################
########################################################################### 
########################################################################### 
print(paste("BEGIN",ctype," [predict] CLUSTER-ing {{o/|o}} k = ",nclust))
fff = predictSubtypeClusterMulti(
       ee,
       measureColumns=colnames(extras),
       myclust,
       clustername = "KMC",
       'eid', 'Years.bl', 0 )
print(table(fff$KMC))
########################################################################### 
########################################################################### 
print("ARGER")
mf=paste("fluid_intelligence_score_f20016~(subjectAge_BL+sex_f31_0_0)*KMC+",gpca)
print(summary(lm(mf,data=fff)))
#######################################
print("BEGIN LONG SEARCH")
pp=getNamesFromDataframe( "", fff, exclusions=c("T1Hier","rsfMRI","DTI","mean_fa","mean_mo","mean_l2","T1w","mean_l3","mean_l1","simlr","bold_effect","brain_position","groupdefined_mask","Years.bl","mean_md_","volume_of_grey_matter",'fa_skel',"genetic_principal_components","weightedmean_isovf","median_t2star","weightedmean_od","t1_brain","_in_tract","in_t1_","fmri_","percentile_of_zstatistic","_SNR","to_standard_space","_mass","_fat","_evr","faceshapes","job_coding",
"time_to_press","triplet_entered","triplet","cervical") )
######################
isbl = fff$Years.bl==0
for ( x in sample(pp) ) {
    if ( class(fff[,x]) == 'numeric' | class(fff[,x]) == 'integer' ) {
        zzz=fff[!is.na(fff[,x]),]
        zzz$yblr = round( zzz$Years.bl)
        tsel = table( zzz$yblr, zzz$eid )
        # eidtbl = table( zzz$eid )
        usesubs='x'
        if ( '2' %in% rownames(tsel) )
            usesubs = intersect( 
                names(tsel['2', ][tsel['2', ]==1 ]),
                names(tsel['0', ][tsel['0', ]==1 ]) )
#        usesubs=intersect( zzz$eid[isbl], zzz$eid[ !isbl ])
#        usesubs=unique( zzz$eid[isbl] )
        if ( length(usesubs)  > 100 ) {
            # check for change
            ischange=FALSE
            for ( k in sample(1:nrow(zzz),44) ) {
                selsub = (zzz$eid == usesubs[k])
                if ( sum(selsub,na.rm=T) > 1 )
                    ischange = ischange | ( var( zzz[ selsub , x ], na.rm=T ) > 0 )
                if ( ischange ) break
                }
            if ( ischange  ) {
                zzz=zzz[zzz$eid %in% usesubs, ]
                zzz$yblr = round( zzz$Years.bl)
                tsel = table( zzz$yblr )
                print( tsel )
                tsel = names(tsel[tsel > 100])
                plotSubtypeChange( zzz[zzz$yblr %in% tsel,], idvar='eid',
                    measurement=x, whiskervar='se',
                    subtype='KMC', vizname='yblr' ) %>% print() 
                myform = paste(x,"~(1|eid)+(subjectAge_BL+sex_f31_0_0)+(", 
                    simnames, ")*Years.bl")
                myform = paste(x,"~(1|eid)+townsend_deprivation_index_at_recruitment_f22189_0_0+",gpca,"+(subjectAge_BL+sex_f31_0_0)+KMC+KMC:Years.bl")
#                myform = paste(x,"~(1|eid)+KMC*Years.bl")
                mdl =lmer( myform,data=zzz)
                print(coefficients(summary( mdl))[,-c(1:3)])
                print(x)
#                visreg::visreg(mdl,'Years.bl',by='KMC')
            }
        }
    }
}



## pd prs
ee=fff[isbl,]
mdl = lm( paste("standard_prs_for_parkinsons_disease_pd_f26260_0_0~(subjectAge_BL+sex_f31_0_0)+", gpca, "+",simnames), data=ee)
mdl = lm( paste("standard_prs_for_alzheimers_disease_ad_f26206_0_0~(subjectAge_BL+sex_f31_0_0)+",simnames), data=ee)
summary( mdl )
visreg::visreg(mdl,'simlrt110')

mdl = lmer( paste("simlrt110~(1|eid)+standard_prs_for_parkinsons_disease_pd_f26260_0_0*Years.bl+(subjectAge_BL+sex_f31_0_0)+", gpca), data=fff)
summary( mdl )
visreg::visreg(mdl,'simlrt110')



## pd prs
ee=fff[isbl,]
x='felt_hated_by_family_member_as_a_child_f20487_0_0'
ee$fhaschild=NA
ee[ fs(ee[,x] == 'Never true'), 'fhaschild' ] = 0
ee[ fs(ee[,x] == 'Rarely true'), 'fhaschild' ] = 1
ee[ fs(ee[,x] == 'Sometimes true'), 'fhaschild' ] = 2
ee[ fs(ee[,x] == 'Very often true'), 'fhaschild' ] = 3
ee[ fs(ee[,x] == 'Often'), 'fhaschild' ] = 4
x='fathers_age_at_death_f1807'
x='spine_bmc_bone_mineral_content_f23312'
x='maximum_digits_remembered_correctly_f20240_0_0'
x='standard_prs_for_parkinsons_disease_pd_f26260_0_0'
x='fhaschild'
x='fluid_intelligence_score_f20016'
zz=ee[!is.na(ee[,x]),]
gpcax='1'
bmdl = lm( paste(x,"~1+",gpcax,"+(subjectAge_BL+sex_f31_0_0)"), data=zz)
mdl = lm( paste(x,"~1+",gpcax,"+(subjectAge_BL+sex_f31_0_0)+(",simnames,")"), data=ee)
mdl = lm( paste(x,"~1+",gpcax,"+(subjectAge_BL+sex_f31_0_0)+(KMC)"), data=ee)
summary( mdl )
pp1=predict(mdl)
ppb=predict(bmdl)
mydf=data.frame(tru=ee[names(pp1),x],img=pp1,base=ppb)
imdl=lm( tru~img,data=mydf)
bmdl=lm( tru~base,data=mydf)
print(paste(rsq(bmdl),rsq(imdl)))
