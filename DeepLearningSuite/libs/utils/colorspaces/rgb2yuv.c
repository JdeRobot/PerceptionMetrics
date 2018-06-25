/*
 *
 *  Copyright (C) 1997-2009 JDE Developers Team
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/. 
 *
 *  Authors : Pablo Miangolarra Tejada <pablo.miangolarra@gmail.com>
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#include "colorspaces.h"

const int MAXIMUN_RGB = 255;

struct YUV * LUT_RGB2YUV [64][64][64];

/* Condicional variable:
 *   0: The table RGB2YUV don't exists.
 *   1: The table RGB2YUV exists.
 */
int isInitTableYUV;

/* mutex */
pthread_mutex_t mutex;

void rgb2yuv_wiki (double r, double g, double b, double *Y, double *U, double *V)
{
 	/* From the wikipedia: yuv color space */

	/* Transform R,G,B to [0,1] range*/

	r = r / MAXIMUN_RGB;
	g = g / MAXIMUN_RGB;
	b = b / MAXIMUN_RGB;

	/*Apply formulas*/

	*Y = r * 0.299 + g * 0.587 + b * 0.114;
	*U = -0.14713*r - 0.289*g + 0.436*b;
	*V =  0.615*r - 0.515*g - 0.100*b;

	//printf(" Valores Y: %.2f U: %.2f V: %.2f ---- R: %.2f G: %.2f B: %.2f\n",*Y,*U,*V,r,g,b);
	
}



void yuv2rgb(double Y, double U, double V, double *r, double *g, double *b)
{
	/* From the wikipedia: yuv color space*/

	*r = Y + 1.13983 * V;
	*g = Y - 0.39465 * U - 0.58060 * V;
	*b = Y + 2.03211 * U;

	if (*r>1.0){*r = 1.0;}
	if (*g>1.0){*g = 1.0;}
	if (*b>1.0){*b = 1.0;}

	if (*r<0.0){*r = 0.0;}
	if (*g<0.0){*g = 0.0;}
	if (*b<0.0){*b = 0.0;}

	/* Return r,g,b in [0,1] range*/

}

/// \brief Function to print unsiged int in binary
void print_status_HSV(unsigned long status)
{

	//const int BITS_PACK = 4;

	unsigned int t = 8;
	unsigned int i;
	unsigned long int j= 1 << (t - 1);

	for (i= t; i > 0; --i)
	{
		printf("%d", (status & j) != 0);
		j>>= 1;

		//if ((i - 1) % BITS_PACK == 0)
		if (i==3)
			printf(" ");
	}

	printf(" (%lu)\n",status);
}


void RGB2YUV_destroyTable ()
{

	int r,g,b;
	int pos_r, pos_g, pos_b;
	int count = 4;

	printf("Destroy Table LUT_RGB2YUV .... OK\n");

	for (b=0;b<=MAXIMUN_RGB;b=b+count)
		for (g=0;g<=MAXIMUN_RGB;g=g+count)
			for (r=0;r<=MAXIMUN_RGB;r=r+count)
			{
				if (r==0) pos_r=0; else pos_r = r/4;
				if (g==0) pos_g=0; else pos_g = g/4;
				if (b==0) pos_b=0; else pos_b = b/4;

				if (LUT_RGB2YUV[pos_r][pos_g][pos_b])
				{
					free(LUT_RGB2YUV[pos_r][pos_g][pos_b]);
				}
			}
	pthread_mutex_lock(&mutex);
	isInitTableYUV = 0;
	pthread_mutex_unlock(&mutex);
}


void RGB2YUV_init()
{
	/* Checking exist one instance */
	pthread_mutex_lock(&mutex);
        if (isInitTableYUV==1)
        {
                pthread_mutex_unlock(&mutex);
                return;
        }
        pthread_mutex_unlock(&mutex);

	printf("Init %s v%s ... \n",NAME,COLORSPACES_VERSION);
	pthread_mutex_lock(&mutex);
	isInitTableYUV = 0;
	pthread_mutex_unlock(&mutex);
}


/// @TODO: Calculate values for create a generic table
void RGB2YUV_createTable()
{

	int r,g,b;
	int count, index;
	int pos_r, pos_g, pos_b;

	struct YUV* newYUV;
	
	/* Checking exist one instance */
	pthread_mutex_lock(&mutex);
	if (isInitTableYUV==1)
	{
		pthread_mutex_unlock(&mutex);
		printf("YUV table already exists\n");
		return;
	}
	pthread_mutex_unlock(&mutex);
	

	count = 4;
	index = 0;

	for (b=0;b<=MAXIMUN_RGB;b=b+count)
		for (g=0;g<=MAXIMUN_RGB;g=g+count)
			for (r=0;r<=MAXIMUN_RGB;r=r+count)
			{
				newYUV = (struct YUV*) malloc(sizeof(struct YUV));
				if (!newYUV)
				{
					printf("Allocated memory error\n");
					exit(-1);
				}

				rgb2yuv_wiki(r,g,b,&(newYUV->Y),&(newYUV->U),&(newYUV->V));

				if (r==0) pos_r=0; else pos_r = r/4;
				if (g==0) pos_g=0; else pos_g = g/4;
				if (b==0) pos_b=0; else pos_b = b/4;

				//printf("[%d,%d,%d] RGB=%d,%d,%d - %.1f,%.1f,%.1f \n",pos_r,pos_g,pos_b,r,g,b,newYUV->H,newYUV->S,newYUV->I);
				LUT_RGB2YUV[pos_r][pos_g][pos_b] = newYUV;

				index++;
			}

	printf("Table 'LUT_RGB2YUV' create with 6 bits (%d values)\n",index);
	
	pthread_mutex_lock(&mutex);
	isInitTableYUV=1;
	pthread_mutex_unlock(&mutex);
}


void RGB2YUV_printYUV (struct YUV* yuv)
{
	printf("YUV: %.1f,%.1f,%.1f\n",yuv->Y,yuv->U,yuv->V);
}


void RGB2YUV_test (void)
{
	int r,g,b;
	const struct YUV* myYUV=NULL;
	struct YUV myYUV2;
	char line[16];
 
	while (1)
	{

		printf("\nIntroduce R,G,B: ");
		fgets(line,16,stdin);
		if ( sscanf(line,"%d,%d,%d",&r,&g,&b)!= 3)
			break;

		myYUV = RGB2YUV_getYUV(r,g,b);
		
		if (myYUV==NULL)
		{
			printf ("Error in myYUV=NULL\n");
			continue;
		}
		
		printf("[Table] RGB: %d,%d,%d -- YUV: %.1f,%.1f,%.1f\n",r,g,b,myYUV->Y,myYUV->U,myYUV->V);

		rgb2yuv_wiki(r,g,b,&myYUV2.Y,&myYUV2.U,&myYUV2.V);

		printf("[Algor] RGB: %d,%d,%d -- YUV: %.1f,%.1f,%.1f\n",r,g,b,myYUV2.Y,myYUV2.U,myYUV2.V);

	}
}

