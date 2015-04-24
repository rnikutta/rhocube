#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <stdio.h>
#include <sys/stat.h>
#include <CCfits>
#include <valarray>
#include <cstring>
#include <vector>

#define RAGGIO_INT 45
#define RAGGIO_EXT 105

using namespace std;

int main(void)
{
	// Quelle di seguito sono le variabili del programma, non toccarle
	int i,j,k,naxis=2,numel=256*256;
	long naxes[]={256,256};
	string s,excdiag;
	double crval[2],r,th,ph,rho;
	valarray<double> el(numel);
	struct stat fileinfo;
	
	// Inserisci tra le virgolette il nome del FITS di output
	const char nome[]="polv0gradi.fits";
	
	
	/**************************************************************************
	* Questa parte costruisce il toro, il modo in cui lo fa non è complicato: *
	* crea un cubo di lato 256 pixel e lo esplora con le variabili i, j e k   *
	* all'interno di questo cubo c'è un toro centrato nel pixel (129,129,129) *
	* i due raggi del toro sono r1=30 e r2=75                                 *
	* per semplicità però utilizzo il valore RAGGIO_EXT=r1+r2=105             *
	* e il valore RAGGIO_INT=r1-r2=45                                         *
	* se vuoi modificarli cambiali da sopra (linee 12 e 13)                   *
	* lungo ogni linea di vista (i e j fissati) per ogni valore di k sommo    *
	* all'elemento di matrice el[i][j] un valore che è 0 se sono fuori dal    *
	* toro oppure 1 se sono dentro. In questo modello assegno inoltre un peso *
	* ad ogni valore proporzionale alla distanza dal centro dal cerchi minore *
	**************************************************************************/	
	
	for(i=0;i<255;i++)
		for(j=0;j<255;j++)
		{
			el[256*i+j]=0; //Inizializzo ogni elemento di matrice a 0
			for(k=0;k<255;k++)
			{
				r=sqrt((129-i)*(129-i)+(129-k)*(129-k)); //Distanza dal centro del toro
				
				if(r>=RAGGIO_INT && r<=RAGGIO_EXT) //Se no sono fuori dal toro
				{
					rho=sqrt((129-j)*(129-j)+(r-(RAGGIO_EXT+RAGGIO_INT)/2.)*(r-(RAGGIO_EXT+RAGGIO_INT)/2.)); //Distanza dal centro del cerchio minore
					if(rho<(RAGGIO_EXT-RAGGIO_INT)/2.) //Se no sono fuori dal toro
						el[256*i+j]+=(RAGGIO_EXT-RAGGIO_INT)/2.-rho; //Peso per la distanza
				}
			}
		}
		
	/*************************************************************************/
	
	
	
	// Attenta che se il file FITS esiste viene sovrascritto!!!
	// Cambia il nome ma non toccare queste due righe
	if(!stat(nome,&fileinfo))
		remove(nome);
	
	
	
	/************************************************************************
	* Questa parte crea il FITS, non dovrebbe essere necessario modificarla *
	************************************************************************/
	
	CCfits::FITS *pfits;
		
	try
	{
		pfits=new CCfits::FITS(nome,DOUBLE_IMG,naxis,naxes);
	}
	catch(CCfits::FITS::CantCreate(excdiag))
	{
		cerr << "Errore 1\n";
		exit(1);
	}
	catch(CCfits::HDU::NoSuchKeyword(excdiag))
	{
		cerr << "Errore 2\n";
		exit(2);
	}
	
	/***********************************************************************/
	
	
	
	// Qui vanno inserite le coordinate del pixel di riferimento
	crval[0]=15*(5+36/60.+43.662/3600);  // RA
	crval[1]=-(69+29/60.+47.59/3600);    // Dec
	////////////////////////////////////////////////////////////
			
			
	
	/*************************************************************************************************
	* Questa parte costruisce l'header del FITS, se vuoi inserire altre parole chiave la sintassi è: *
	* pfits->pHDU().addKey("NOME_PAROLA_CHIAVE",VALORE,"COMMENTO");                                  *
	* Attenta che se VALORE è una stringa bisogna includerla tra "..."                               *
	* Se non vuoi inserire un commento, come terzo argomento metti ""                                *
	*************************************************************************************************/
	
	pfits->pHDU().addKey("CRVAL1",crval[0],"Right Ascension (degrees) of reference pixel");//RA pixel di riferimento
	pfits->pHDU().addKey("CRVAL2",crval[1],"Declination of reference pixel");              //Dec pixel di riferimento
	pfits->pHDU().addKey("EQUINOX",2000.,"J2000 equinox");                                 //Equinozio
	pfits->pHDU().addKey("CTYPE1","RA---SIN","");                                          //Proiezione asse RA
	pfits->pHDU().addKey("CDELT1",-1.2592592592592591e-05,"");                             //Dimensione in RA del pixel di riferimento (in gradi!!)
	pfits->pHDU().addKey("CRPIX1",1.290000000000E+02,"");                                  //Pixel di riferimento (non toccare)
	pfits->pHDU().addKey("CROTA1",0.00000000000E+00,"");                                   //Rotazione dell'asse RA (senso orario, in gradi!!)
	pfits->pHDU().addKey("CUNIT1","deg     ","");                                          //Appunto, le unità sono in gradi!!
	pfits->pHDU().addKey("CTYPE2","DEC--SIN","");                                          //Proiezione asse Dec
	pfits->pHDU().addKey("CDELT2",1.2592592592592591e-05,"");                              //Dimensione in Dec del pixel di riferimento (in gradi!!)
	pfits->pHDU().addKey("CRPIX2",1.290000000000E+02,"");                                  //Pixel di riferimento (non toccare)
	pfits->pHDU().addKey("CROTA2",0.000000000000E+00,"");                                  //Rotazione dell'asse Dec (senso orario, in gradi!!)
	pfits->pHDU().addKey("CUNIT2","deg     ","");                                          //Di nuovo, le unità sono in gradi!!
	
	/************************************************************************************************/
	
	
	
	// Se vuoi inserire un commento nell'header fai così
	pfits->pHDU().writeComment("2012 - Adriano Ingallinera with CCfits");
	
	
	// Questa riga scrive i dati nel FITS, non rimuoverla!!!
	pfits->pHDU().write(1,numel,el);

	return 0;
}

