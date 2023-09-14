library( subtyper )
library( ANTsR )
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
    rsfdd = antsrimpute(scale( dd[,rsfnames] ))
    dtidd = antsrimpute(scale( dd[,dtinames] ))
    t1dd = antsrimpute(scale( dd[,t1names] ))
    xsel = dd$eid %in% crosssubs
    write.csv( rsfdd[xsel,], "/tmp/ukrsfx.csv", row.names=FALSE)
    write.csv( dtidd[xsel,], "/tmp/ukdtix.csv", row.names=FALSE)
    write.csv( t1dd[xsel,], "/tmp/ukt1x.csv", row.names=FALSE)
}
#### longitudinal
lsel = dd$eid %in% longsubs
write.csv( rsfdd[lsel,], "/tmp/ukrsfl.csv", row.names=FALSE)
write.csv( dtidd[lsel,], "/tmp/ukdtil.csv", row.names=FALSE)
write.csv( t1dd[lsel,], "/tmp/ukt1l.csv", row.names=FALSE)

ev0=read.csv("/tmp/ukt1ev.csv")[,-1]
ev1=read.csv("/tmp/ukdtiev.csv")[,-1]
ev2=read.csv("/tmp/ukrsfev.csv")[,-1]

#
pt1=data.frame(data.matrix(t1dd[lsel,]) %*% t(data.matrix(ev0)))
colnames(pt1)=paste0("t1",1:ncol(pt1))
pdti=data.frame(data.matrix(dtidd[lsel,]) %*% t(data.matrix(ev1)))
colnames(pdti)=paste0("dti",1:ncol(pt1))
prsf=data.frame(data.matrix(rsfdd[lsel,]) %*% t(data.matrix(ev2)))
colnames(prsf)=paste0("rsf",1:ncol(pt1))


library(lmerTest)
ee=cbind(dd[lsel,],pt1,pdti,prsf)
nsim=ncol(pt1)
simnames=paste(
    paste0(paste0("t1",1:nsim),collapse='+'), "+",
    paste0(paste0("dti",1:nsim),collapse='+'),"+",
    paste0(paste0("rsf",1:nsim),collapse='+')
)
mf=paste("age_MRI~(1|eid)+sex_f31_0_0+",simnames)
print(summary(lmer(mf,data=ee)))


intnames=getNamesFromDataframe("luid",dd)
for ( i in intnames) {
    print(i)
    print( table( !is.na( ee[,i]) ))
}

mf=paste("fluid_intelligence_score_f20016~(1|eid)+age_MRI+sex_f31_0_0+",simnames)
print(summary(lmer(mf,data=ee)))


# some clustering
#
pt1=data.frame(data.matrix(t1dd[xsel,]) %*% t(data.matrix(ev0)))
colnames(pt1)=paste0("t1",1:nsim)
pdti=data.frame(data.matrix(dtidd[xsel,]) %*% t(data.matrix(ev1)))
colnames(pdti)=paste0("dti",1:nsim)
prsf=data.frame(data.matrix(rsfdd[xsel,]) %*% t(data.matrix(ev2)))
colnames(prsf)=paste0("rsf",1:nsim)
nsimu=3
extras=cbind(pt1[,1:nsimu],pdti[,1:nsimu],prsf[,1:nsimu])
ee=cbind(dd[xsel,],extras)
nclust = 4
#
myclust = trainSubtypeClusterMulti(
       ee,
       measureColumns=colnames(extras),
       method = "kmeansflex",
       nclust  )


pt1l=data.frame(data.matrix(t1dd[lsel,]) %*% t(data.matrix(ev0)))
colnames(pt1l)=paste0("t1",1:ncol(pt1))
pdtil=data.frame(data.matrix(dtidd[lsel,]) %*% t(data.matrix(ev1)))
colnames(pdtil)=paste0("dti",1:ncol(pt1))
prsfl=data.frame(data.matrix(rsfdd[lsel,]) %*% t(data.matrix(ev2)))
colnames(prsfl)=paste0("rsf",1:ncol(pt1))
ff=cbind(dd[lsel,],pt1l,pdtil,prsfl)

fff = predictSubtypeClusterMulti(
       ff,
       measureColumns=colnames(extras),
       myclust,
       clustername = "KMC",
       'eid', 'Years.bl', 0 )
     
#

pp=getNamesFromDataframe( "", fff, exclusions=c("T1Hier","rsfMRI","DTI","mean_fa","mean_mo","mean_l2","T1w","mean_l3","mean_l1") )

isbl = fff$Years.bl==0
for ( x in sample(pp) ) {
    if ( class(fff[,x]) == 'numeric' ) {
        zzz=fff[!is.na(fff[,x]),]
        usesubs=intersect( zzz$eid[isbl], zzz$eid[ !isbl ])
        if ( length(usesubs)  > 1000 ) {
            # check for change
            ischange=FALSE
            for ( k in 1:100 )
                ischange = ischange | var( zzz[ zzz$eid == usesubs[k], x ] > 0 )
            if ( ischange ) {
                zzz=zzz[zzz$eid %in% usesubs, ]
                print(x)
                zzz$yblr = round( zzz$Years.bl*2)
                tsel = table( zzz$yblr )
                tsel = names(tsel[tsel > 50])
                plotSubtypeChange( zzz[zzz$yblr %in% tsel,], idvar='eid',
                    measurement=x, 
                    subtype='KMC', vizname='yblr' ) %>% print() 
            }
        }
    }
}
