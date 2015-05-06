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

#define RAGGIO_INT0 10
#define RAGGIO_EXT0 5

#define RAGGIO_INT 10
#define RAGGIO_EXT 15
#define RAGGIO_INTbis 10
#define RAGGIO_EXTbis 25
#define RAGGIO_INT2 10 
#define RAGGIO_EXT2 35
#define RAGGIO_INT3bis 10 
#define RAGGIO_EXT3bis 25
#define RAGGIO_INT3 10 
#define RAGGIO_EXT3 15
#define RAGGIO_INT4 10 
#define RAGGIO_EXT4 5
#define RAGGIO_EXT5 30

using namespace std;

int main(void)
{
	// Quelle di seguito sono le variabili del programma, non toccarle
	int i,j,k,naxis=2,numel=256*256;
	long naxes[]={256,256};
	string s,excdiag;
	double crval[2],r,th,ph,rho,rho0,rhobis,rho2,rho3bis,rho3,rho4,rho5;
	valarray<double> el(numel);
	struct stat fileinfo;
	
	// Inserisci tra le virgolette il nome del FITS di output
	const char nome[]="modelloR127.fits";
	
	
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

				if(r>=RAGGIO_INT0 && r<=RAGGIO_EXT0)//Se no sono fuori dal toro
				{
					rho0=sqrt((99-j)*(99-j)+(r-(RAGGIO_EXT0+RAGGIO_INT0)/2.)*(r-(RAGGIO_EXT0+RAGGIO_INT0)/2.)); //Distanza dal centro del cerchio minore
					if(rho0<(RAGGIO_EXT0-RAGGIO_INT0)/2.) //Se no sono fuori dal toro
						el[256*i+j]+=(RAGGIO_EXT0-RAGGIO_INT0)/2.-rho0; //Peso per la distanza
				}
				
				if(r>=RAGGIO_INT && r<=RAGGIO_EXT)//Se no sono fuori dal toro
				{
					rho=sqrt((109-j)*(109-j)+(r-(RAGGIO_EXT+RAGGIO_INT)/2.)*(r-(RAGGIO_EXT+RAGGIO_INT)/2.)); //Distanza dal centro del cerchio minore
					if(rho<(RAGGIO_EXT-RAGGIO_INT)/2.) //Se no sono fuori dal toro
						el[256*i+j]+=(RAGGIO_EXT-RAGGIO_INT)/2.-rho; //Peso per la distanza
				}
				if(r>=RAGGIO_INTbis && r<=RAGGIO_EXTbis)//Se no sono fuori dal toro
				{
					rhobis=sqrt((119-j)*(119-j)+(r-(RAGGIO_EXTbis+RAGGIO_INTbis)/2.)*(r-(RAGGIO_EXTbis+RAGGIO_INTbis)/2.)); //Distanza dal centro del cerchio minore
					if(rhobis<(RAGGIO_EXTbis-RAGGIO_INTbis)/2.) //Se no sono fuori dal toro
						el[256*i+j]+=(RAGGIO_EXTbis-RAGGIO_INTbis)/2.-rhobis; //Peso per la distanza
				}


				if(r>=RAGGIO_INT2 && r<=RAGGIO_EXT2)//Se no sono fuori dal toro
				{	
					rho2=sqrt((129-j)*(129-j)+(r-(RAGGIO_EXT2+RAGGIO_INT2)/2.)*(r-(RAGGIO_EXT2+RAGGIO_INT2)/2.)); //Distanza dal centro del cerchio minore
					if(rho2<(RAGGIO_EXT2-RAGGIO_INT2)/2.) //Se no sono fuori dal toro
						el[256*i+j]+=(RAGGIO_EXT2-RAGGIO_INT2)/2.-rho2; //Peso per la distanza
				}
				if(r>=RAGGIO_INT3bis && r<=RAGGIO_EXT3bis) //Se no sono fuori dal toro
				{	
					rho3bis=sqrt((139-j)*(139-j)+(r-(RAGGIO_EXT3bis+RAGGIO_INT3bis)/2.)*(r-(RAGGIO_EXT3bis+RAGGIO_INT3bis)/2.)); //Distanza dal centro del cerchio minore
					if(rho3bis<(RAGGIO_EXT3bis-RAGGIO_INT3bis)/2.) //Se no sono fuori dal toro
						el[256*i+j]+=(RAGGIO_EXT3bis-RAGGIO_INT3bis)/2.-rho3bis; //Peso per la distanza
				}
				if(r>=RAGGIO_INT3 && r<=RAGGIO_EXT3) //Se no sono fuori dal toro
				{	
					rho3=sqrt((149-j)*(149-j)+(r-(RAGGIO_EXT3+RAGGIO_INT3)/2.)*(r-(RAGGIO_EXT3+RAGGIO_INT3)/2.)); //Distanza dal centro del cerchio minore
					if(rho3<(RAGGIO_EXT3-RAGGIO_INT3)/2.) //Se no sono fuori dal toro
						el[256*i+j]+=(RAGGIO_EXT3-RAGGIO_INT3)/2.-rho3; //Peso per la distanza
				}
				if(r>=RAGGIO_INT4 && r<=RAGGIO_EXT4) //Se no sono fuori dal toro
				{	
					rho4=sqrt((159-j)*(159-j)+(r-(RAGGIO_EXT4+RAGGIO_INT4)/2.)*(r-(RAGGIO_EXT4+RAGGIO_INT4)/2.)); //Distanza dal centro del cerchio minore
					if(rho4<(RAGGIO_EXT4-RAGGIO_INT4)/2.) //Se no sono fuori dal toro
						el[256*i+j]+=(RAGGIO_EXT4-RAGGIO_INT4)/2.-rho4; //Peso per la distanza
				}
				if(r<=RAGGIO_EXT5) //Se no sono fuori dal toro
				{	
					rho5=sqrt((129-j)*(129-j)+(r)*(r)); //Distanza dal centro del cerchio minore
					if(rho5<(RAGGIO_EXT5)) //Se no sono fuori dal cerchio
						el[256*i+j]+=rho5- (RAGGIO_EXT5/2); //Peso per la distanza
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
	pfits->pHDU().writeComment("2015 - Claudia - da Adriano Ingallinera routine with CCfits");
	
	
	// Questa riga scrive i dati nel FITS, non rimuoverla!!!
	pfits->pHDU().write(1,numel,el);

	return 0;
}

