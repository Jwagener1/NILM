#include <stdint.h>
#include <stdlib.h>
#include <pru_cfg.h>
#include <pru_ctrl.h>
#include <stddef.h>
#include <rsc_types.h>
#include "resource_table_empty.h"
#include "init_pins_empty.h"
#include "prugpio.h"

volatile register unsigned int __R30;
volatile register unsigned int __R31;

#define SHARED_RAM_ADDRESS 0x10000
#define d1 315
#define d2 480
#define d3 145
#define d4 90

// #define d1 730
// #define d2 960
// #define d3 145
// #define d4 90
unsigned int volatile __far * const SHARED_RAM = (unsigned int *) (SHARED_RAM_ADDRESS);
unsigned int temp;
unsigned int temp1;

int i = 0;


void main(void) {
	//CT_CFG.SYSCFG_bit.STANDBY_INIT = 0;
	
	uint32_t cs   = P9_16;
	uint32_t clk  = P9_14;
	
	for (i = 0; i < 1024; i++)
	{
    	__R30 |= clk;
		__delay_cycles(100/5); 
		__R30 &= ~clk;
		__delay_cycles(100/5);
	}
	__delay_cycles(10000/5);

	while(1)
	
	{
	__R30 |= cs;					//
	temp1 = 0;
	temp1 |= ((0x80000000 & temp));
	temp1 |= ((0x20000000 & temp)<<1);
	temp1 |= ((0x08000000 & temp)<<2);
	temp1 |= ((0x02000000 & temp)<<3);
	temp1 |= ((0x00800000 & temp)<<4);
	temp1 |= ((0x00200000 & temp)<<5);
	temp1 |= ((0x00080000 & temp)<<6);
	temp1 |= ((0x00020000 & temp)<<7);
	temp1 |= ((0x00008000 & temp)<<8);
	temp1 |= ((0x00002000 & temp)<<9);
	temp1 |= ((0x00000800 & temp)<<10);
	temp1 |= ((0x00000200 & temp)<<11);
	temp1 |= ((0x00000080 & temp)<<12);
	temp1 |= ((0x00000020 & temp)<<13);
	temp1 |= ((0x00000008 & temp)<<14);
	temp1 |= ((0x00000002 & temp)<<15);
	
	temp1 |= ((0x40000000 & temp)>>15);
	temp1 |= ((0x10000000 & temp)>>14);
	temp1 |= ((0x04000000 & temp)>>13);
	temp1 |= ((0x01000000 & temp)>>12);
	temp1 |= ((0x00400000 & temp)>>11);
	temp1 |= ((0x00100000 & temp)>>10);
	temp1 |= ((0x00040000 & temp)>>9);
	temp1 |= ((0x00010000 & temp)>>8);
	temp1 |= ((0x00004000 & temp)>>7);
	temp1 |= ((0x00001000 & temp)>>6);
	temp1 |= ((0x00000400 & temp)>>5);
	temp1 |= ((0x00000100 & temp)>>4);
	temp1 |= ((0x00000040 & temp)>>3);
	temp1 |= ((0x00000010 & temp)>>2);
	temp1 |= ((0x00000004 & temp)>>1);
	temp1 |= ((0x00000001 & temp));
	

	__delay_cycles(100/5);          
	__R30 &= ~cs;
		
		
	__delay_cycles(1);
	*SHARED_RAM = temp1;
	__delay_cycles(2360/5); 
	temp = 0x00000000;
	
		
		
	//1
	__R30 |= clk;
	__delay_cycles(d2/5); 
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<29);
	__delay_cycles(d1/5); 
	//2
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<27);
	__delay_cycles(d1/5); 
	//3
	__R30 |= clk;
	__delay_cycles(d2/5);	 
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<25);
	__delay_cycles(d1/5); 		
	//4
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<23);
	__delay_cycles(d1/5); 
	//5
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<21);
	__delay_cycles(d1/5); 
	//6
	__R30 |= clk;
	__delay_cycles(d2/5);	
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<19);
	__delay_cycles(d1/5);
	//7
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<17);
	__delay_cycles(d1/5);
	//8
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<15);
	__delay_cycles(d1/5);
	//9
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<13);
	__delay_cycles(d1/5);
	//10
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<11);
	__delay_cycles(d1/5);
	//11
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<9);
	__delay_cycles(d1/5);
	//12
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<7);
	__delay_cycles(d1/5);
	//13
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<5);
	__delay_cycles(d1/5);
	//14
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<3);
	__delay_cycles(d1/5);
	//15
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)<<1);
	__delay_cycles(d1/5);
	//16
	__R30 |= clk;
	__delay_cycles(d2/5);
	__R30 &= ~clk;
	__delay_cycles(d3/5); 
	temp |= ((0x00000006 & __R31)>>1);
    __delay_cycles(d1*2/5);
    __delay_cycles(d4/5);
	}
}

