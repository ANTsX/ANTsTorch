library( subtyper )
library( ANTsR )
ukbfn = "data/allcsvs/ukb672504_imaging_long.csv"
if ( ! exists("dd") ) {
    dd=read.csv(ukbfn)
    dd=dd[ fs( dd$T1Hier_resnetGrade > 1 ), ]
    dd=dd[ !is.na(dd$DTI_dti_FD_mean) & !is.na( dd$rsfMRI_rsf_FD_mean ),  ]
}
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
#### longitudinal
lsel = dd$eid %in% longsubs
write.csv( rsfdd[lsel,], "/tmp/ukrsfl.csv", row.names=FALSE)
write.csv( dtidd[lsel,], "/tmp/ukdtil.csv", row.names=FALSE)
write.csv( t1dd[lsel,], "/tmp/ukt1l.csv", row.names=FALSE)
