#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>

void delay(int milliseconds);


void data_parse(volatile int dataram, int16_t* vol, int16_t* cur);

int16_t two_comp_to_dec(int16_t input);

double voltage_conversion(int16_t vol);

double current_conversion(int16_t cur);

void c_sample(volatile int* pru_res,int n, double *voltage,double *current);


#define PRUSS_SHARED_RAM_OFFSET		0x10000

void c_square(int n, double *array_in, double *array_out,double *array_out2)
{ //return the square of array_in of length n in array_out
    int i;
    

    for (i = 0; i < n; i++)
    {
        array_out[i] = array_in[i] * array_in[i];
        array_out2[i] = array_out[i] * array_out[i];
    }
}


volatile int* mem_init()
{
    int argc;
    const char** argv;
    unsigned int mem_dev;
	char *shared;
	/* Allocate shared memory pointer to PRU0 DATARAM */
	if(argc==2)
		mem_dev = open(argv[1], O_RDWR | O_SYNC);
	else
		mem_dev = open("/dev/uio0", O_RDWR | O_SYNC);
	volatile int *shared_dataram = mmap(NULL,
		16+PRUSS_SHARED_RAM_OFFSET,	/* grab 16 bytes of shared dataram, must allocate with offset of 0 */
		PROT_READ | PROT_WRITE ,
		MAP_SHARED,
		mem_dev,
		0);
	shared_dataram += (PRUSS_SHARED_RAM_OFFSET/4);
	printf("shared_dataram = %p\n", shared_dataram);
	return shared_dataram;
}

int print_val(volatile int* value)
{
    int temp = *value;
    printf("PRU Value in Hex : 0x%.8X\n", temp);
    printf("PRU Value in Hex : %d\n", temp);
    return temp;
}

void c_sample(volatile int* pru_res,int n, double *voltage,double *current)
{ 
    int i = 0;
    int check = 0;
    
    int16_t voltage_2_complement = 0;
    int16_t current_2_complement = 0;

    int16_t voltage_decimal = 0;
    int16_t current_decimal = 0;
    
    
    
    while (i < n)
    {
        if (*pru_res != check)
	    {
        data_parse(*pru_res, &voltage_2_complement, &current_2_complement);
        voltage_decimal = two_comp_to_dec(voltage_2_complement);
        voltage[i] = voltage_conversion(voltage_decimal);
        data_parse(*pru_res, &voltage_2_complement, &current_2_complement);
        current_decimal = two_comp_to_dec(current_2_complement);
        current[i] = current_conversion(current_decimal);
        i++;
        check = *pru_res;
	    }
    }
    
}

void data_parse(volatile int dataram, int16_t* vol, int16_t* cur)
{
    *vol = ((0xFFFF0000 & dataram) >> 16);
    *cur = (0x0000FFFF & dataram);
    
}

int16_t two_comp_to_dec(int16_t input)
{
    
    if ((0x8000 & input) == 0)
    {
        return input;
    }
    else
    {
        return -(~input + 1);
    }
}

double voltage_conversion(int16_t vol)
{
    double t = ((double) vol / 32767.0) * 5.000;
    return t;
}

double current_conversion(int16_t cur)
{
    double t = ((double) cur / 32767.0) * 5.0;
    return t;
}

void delay(int milliseconds)
{
    long pause;
    clock_t now,then;

    pause = milliseconds*(CLOCKS_PER_SEC/1000);
    now = then = clock();
    while( (now-then) < pause )
        now = clock();
}


