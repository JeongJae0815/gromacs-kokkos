/*
 *       $Id$
 *
 *       This source code is part of
 *
 *        G   R   O   M   A   C   S
 *
 * GROningen MAchine for Chemical Simulations
 *
 *            VERSION 2.0
 * 
 * Copyright (c) 1991-1997
 * BIOSON Research Institute, Dept. of Biophysical Chemistry
 * University of Groningen, The Netherlands
 * 
 * Please refer to:
 * GROMACS: A message-passing parallel molecular dynamics implementation
 * H.J.C. Berendsen, D. van der Spoel and R. van Drunen
 * Comp. Phys. Comm. 91, 43-56 (1995)
 *
 * Also check out our WWW page:
 * http://rugmd0.chem.rug.nl/~gmx
 * or e-mail to:
 * gromacs@chem.rug.nl
 *
 * And Hey:
 * Green Red Orange Magenta Azure Cyan Skyblue
 */
static char *SRCID_genbox_c = "$Id$";

#include "sysstuff.h"
#include "typedefs.h"
#include "smalloc.h"
#include "string2.h"
#include "physics.h"
#include "confio.h"
#include "copyrite.h"
#include "math.h"
#include "macros.h"
#include "random.h"
#include "futil.h"
#include "vdw.h"
#include "names.h"
#include "pbc.h"
#include "fatal.h"
#include "statutil.h"
#include "vec.h"
#include "gstat.h"
#include "addconf.h"

typedef struct {
  char **name;
  rvec x;
  rvec v;
  real r;
} atom_;

typedef struct {
  char  **name;
  int   nr_atoms;
  atom_ *atom;
} molecule_;


void print_stat(rvec *x,int natoms,matrix box)
{
  int i,m;
  rvec xmin,xmax;
  for(m=0;(m<DIM);m++) {
    xmin[m]=x[0][m];
    xmax[m]=x[0][m];
  }
  for(i=0;(i<natoms);i++) {
    for (m=0;m<DIM;m++) {
      xmin[m]=min(xmin[m],x[i][m]);
      xmax[m]=max(xmax[m],x[i][m]);
    }   
  }
  for(m=0;(m<DIM);m++)
    fprintf(stderr,"DIM %d XMIN %8.3f XMAX %8.3f BOX %8.3f\n",m,xmin[m],xmax[m],box[m][m]);
}

static real min_distance(char resname[],char atomname[],
			 t_vdw **vdw,int *max_i,real r_distance)
{
  int i,match=0;
  real distance=0;
  for (i=0; ((i<*max_i) && (match < 1)); i++ )
    if ( strcmp(atomname,(*vdw)[i].atomname) == 0){
      distance = (*vdw)[i].distance;
      match=1;
    } 
  if (match==0) {
    (*max_i)++;
    srenew(*vdw,*max_i);
    (*vdw)[*max_i-1].distance=r_distance;
    strcpy((*vdw)[*max_i-1].atomname,atomname);
    distance = (*vdw)[*max_i-1].distance;
    fprintf(stderr,"distance of %s %s is set to %f\n",
	    resname,(*vdw)[*max_i-1].atomname,
	    (*vdw)[*max_i-1].distance);
  }
  return distance;
} /*min_distance()*/

static void mk_vdw(t_atoms *a,real rvdw[],
		   t_vdw **vdw,int *max_vdw,
		   real r_distance)
{
  int i;
  
  /*initialise van der waals arrays of configuration */
  fprintf(stderr,"Initialising van der waals distances...\n");
  for(i=0; (i < a->nr); i++)
    rvdw[i]=min_distance(*(a->resname[a->atom[i].resnr]),
			 *(a->atomname[i]),
			 vdw,max_vdw,r_distance);
}

/*m_swap swaps two molecules i and j m_swap is used by m_qsort*/
void m_swap(molecule_ *molecule,int i,int j)
{
  molecule_ temp;
  temp = molecule[i];
  molecule[i] = molecule[j];
  molecule[j] = temp;
} /*m_swap()*/

static int comp_mol(const void *a,const void *b)
{
  molecule_ *ma,*mb;

  ma=(molecule_ *)a;
  mb=(molecule_ *)b;

  return strcmp(*ma->name,*mb->name);
}

/*m_qsort() is a standard quicksort() routine adapted to the molecule_ 
 structrure*/
void m_qsort(molecule_ *molecule, int left, int right)
{
  int i,last;
  if (left>=right)
    return;
  m_swap(molecule,left,(left+right)/2);
  last = left;
  for (i=left+1;i<=right;i++)
    if(strcmp(*molecule[i].name,*molecule[left].name) < 0)
      m_swap(molecule,++last,i);
  m_swap(molecule,left,last);
  m_qsort(molecule,left,last-1);
  m_qsort(molecule,last+1,right);
} /*m_qsort()*/


/*m_sort() sorts the structure atoms by molecule name*/
void m_sort(t_atoms *atoms,rvec *x,rvec *v,real *r,int left, int right)
{
  int k=0,l=0,i,j,nr;
  molecule_ *molecule;
  /*print message*/
  fprintf(stderr,"Sorting configuration\n");
  /*the structure atoms is transferred to to structure molecule_*/
  snew(molecule,atoms->nres);
  for (i=0;(i<atoms->nr);i++) {
    if (i>0)
      if (atoms->atom[i].resnr != atoms->atom[i-1].resnr) {
	l=0;
        k++;
	if (k!=atoms->atom[i].resnr)
	  printf("big error");
      }
    molecule[k].name = atoms->resname[atoms->atom[i].resnr];
    srenew(molecule[k].atom,l+1);
    molecule[k].atom[l].name=atoms->atomname[i];
    molecule[k].atom[l].x[0]=x[i][0];
    molecule[k].atom[l].x[1]=x[i][1];
    molecule[k].atom[l].x[2]=x[i][2];
    molecule[k].atom[l].v[0]=v[i][0];
    molecule[k].atom[l].v[1]=v[i][1];
    molecule[k].atom[l].v[2]=v[i][2];
    molecule[k].atom[l].r=r[i];
    l++;
    molecule[k].nr_atoms=l;
  }

  /*perform a quicksort*/
  /* m_qsort(molecule,left,right); */
  qsort(molecule,atoms->nres,sizeof(molecule[0]),comp_mol);

  /*transfer the structure molecule_ to atoms*/
  for (i=0,nr=0;(i<atoms->nres);i++) {
    for(j=0;(j<molecule[i].nr_atoms);j++) {
      atoms->resname[i]=molecule[i].name;
      atoms->atomname[nr]=molecule[i].atom[j].name;
      atoms->atom[nr].resnr=i;    
      x[nr][0]=molecule[i].atom[j].x[0];
      x[nr][1]=molecule[i].atom[j].x[1];
      x[nr][2]=molecule[i].atom[j].x[2];
      v[nr][0]=molecule[i].atom[j].v[0];
      v[nr][1]=molecule[i].atom[j].v[1];
      v[nr][2]=molecule[i].atom[j].v[2];
      r[nr]=molecule[i].atom[j].r;
      nr++;
    }
  }
}/*m_sort()*/

char *add_mol3(char *mol3,int nmol_3,int seed,int ntb,
	       t_atoms *atoms_1,rvec **x_1,rvec **v_1,real **r_1,matrix box_1,
	       t_vdw **vdw,int *max_vdw,real r_distance)
{
  static  char    *title_3;
  t_atoms atoms_3;
  rvec    *x_3,*v_3,*x_n,*v_n;
  real    *r_3;
  matrix  box_3;
  int     i,mol,onr_1;
  real    alfa_3,beta_3;
  rvec    offset_x;
  int     try;
  
  /*read number of atoms of configuration 3*/
  get_stx_coordnum(mol3,&atoms_3.nr);
  if (atoms_3.nr == 0) {
    fprintf(stderr,"No molecule in %s, please check your input\n",mol3);
    exit(1);
  }
  /*allocate memory for atom coordinates of configuration 3*/
  snew(x_3,atoms_3.nr);
  snew(v_3,atoms_3.nr);
  snew(r_3,atoms_3.nr);
  snew(atoms_3.resname,atoms_3.nr);
  snew(atoms_3.atomname,atoms_3.nr);
  snew(atoms_3.atom,atoms_3.nr);
  snew(x_n,atoms_3.nr);
  snew(v_n,atoms_3.nr);
  snew(title_3,STRLEN);
  
  /*read residue number, residue names, atomnames, coordinates etc.*/
  fprintf(stderr,"Reading molecule configuration \n");
  read_stx_conf(mol3,title_3,&atoms_3,x_3,v_3,box_3);
  fprintf(stderr,"%s\nContaining %d atoms in %d residue\n",
	  title_3,atoms_3.nr,atoms_3.nres);
  srenew(atoms_3.resname,atoms_3.nres);  
    
  /*initialise van der waals arrays of configuration 3*/
  mk_vdw(&atoms_3,r_3,vdw,max_vdw,r_distance);

  try=mol=0;
  while ((mol < nmol_3) && (try < 10*nmol_3)) {
    fprintf(stderr,"\rTry %d",try++);
    srenew(atoms_1->resname,(atoms_1->nres+nmol_3));
    srenew(atoms_1->atomname,(atoms_1->nr+atoms_3.nr*nmol_3));
    srenew(atoms_1->atom,(atoms_1->nr+atoms_3.nr*nmol_3));
    for (i=0;(i<atoms_3.nr);i++) {
      if (atoms_3.atom[i].resnr!=0) 
	fatal_error(0,"more then one residue in configuration 3\n"
		    "program terminated\n");
      srenew(atoms_1->resname,(atoms_1->nres+nmol_3));
      srenew(atoms_1->atomname,(atoms_1->nr+atoms_3.nr*nmol_3));
      srenew(atoms_1->atom,(atoms_1->nr+atoms_3.nr*nmol_3));
      srenew(*x_1,(atoms_1->nr+atoms_3.nr*nmol_3));
      srenew(*v_1,(atoms_1->nr+atoms_3.nr*nmol_3));
      srenew(*r_1,(atoms_1->nr+atoms_3.nr*nmol_3));
      x_n[i][XX]=x_3[i][XX];
      x_n[i][YY]=x_3[i][YY];
      x_n[i][ZZ]=x_3[i][ZZ];
      v_n[i][XX]=v_3[i][XX];
      v_n[i][YY]=v_3[i][YY];
      v_n[i][ZZ]=v_3[i][ZZ];
    }
    alfa_3=2*M_PI*rando(&seed);
    beta_3=2*M_PI*rando(&seed);
    rotate_conf(atoms_3.nr,x_n,v_n,alfa_3,beta_3,0);
    offset_x[XX]=box_1[XX][XX]*rando(&seed);
    offset_x[YY]=box_1[YY][YY]*rando(&seed);
    offset_x[ZZ]=box_1[ZZ][ZZ]*rando(&seed);
    gen_box(0,atoms_3.nr,x_n,box_3,offset_x,TRUE);
    if ((in_box(ntb,box_1,x_n[0]) != 0) || 
	(in_box(ntb,box_1,x_n[atoms_3.nr-1])!=0))
      continue;
    onr_1=atoms_1->nr;
     
    add_conf(atoms_1,*x_1,*v_1,*r_1,ntb,box_1,
	     &atoms_3,x_n,v_n,r_3,FALSE,1);
    if (atoms_1->nr==(atoms_3.nr+onr_1)) {
      mol++;
      fprintf(stderr," success (now %d atoms)!\n",atoms_1->nr);
    }
  }
  /*print number of molecules added*/
  fprintf(stderr,"Added %d molecules (out of %d requested) of %s\n",
	  mol,nmol_3,*atoms_3.resname[0]); 
    
  return title_3;
}

void add_solv(char *solv,int ntb,
	      t_atoms *atoms_1,rvec **x_1,rvec **v_1,real **r_1,matrix box_1,
	      t_vdw **vdw,int *max_vdw,real r_distance,
	      int *atoms_added,int *residues_added,int maxmol)
{
  int     i,d,n,nmol;
  ivec    n_box;
  char    filename[STRLEN];
  char    title_2[STRLEN];
  t_atoms atoms_2;
  rvec    *x_2,*v_2;
  real    *r_2;
  matrix  box_2;
  int     onr_1,onres_1;
  rvec    xmin;

  strncpy(filename,libfn(solv),STRLEN);
  get_stx_coordnum(filename,&atoms_2.nr); 
  if (atoms_2.nr == 0) {
    fprintf(stderr,"No solvent in %s, please check your input\n",filename);
    exit(1);
  }
  snew(x_2,atoms_2.nr);
  snew(v_2,atoms_2.nr);
  snew(r_2,atoms_2.nr);
  snew(atoms_2.resname,atoms_2.nr);
  snew(atoms_2.atomname,atoms_2.nr);
  snew(atoms_2.atom,atoms_2.nr);
  fprintf(stderr,"Reading solvent configuration \n");
  read_stx_conf(filename,title_2,&atoms_2,x_2,v_2,box_2);
  fprintf(stderr,"%s\nContaining %d atoms in %d residues\n",
	  title_2,atoms_2.nr,atoms_2.nres);
/*   srenew(atoms_2.resname,atoms_2.nres); */
  
  /*initialise van der waals arrays of configuration 2*/
  mk_vdw(&atoms_2,r_2,vdw,max_vdw,r_distance);
  
  /*calculate the box scaling factors n_box[0...DIM]*/
  nmol=1;
  for (i=0; (i < DIM);i++) {
    n_box[i]=(int)(box_1[i][i]/box_2[i][i])+1;
    nmol*=n_box[i];
  }
  fprintf(stderr,"\nsolvent configuration multiplicated %d times\n",nmol);  
  
  /*realloc atoms_2 for the new configuration*/
  srenew(atoms_2.resname,atoms_2.nres*nmol*3);
  srenew(atoms_2.atomname,atoms_2.nr*nmol*3);
  srenew(atoms_2.atom,atoms_2.nr*nmol*3);
  srenew(x_2,atoms_2.nr*nmol*3);
  srenew(v_2,atoms_2.nr*nmol*3);
  srenew(r_2,atoms_2.nr*nmol*3);
  
  /*generate a new configuration 2*/
  genconf(&atoms_2,x_2,v_2,r_2,box_2,n_box);

#ifdef DEBUG
  print_stat(x_2,atoms_2.nr,box_2);
#endif
  /* move the generated configuration to the positive octant */
  for(d=0;(d<DIM);d++)
    xmin[d]=x_2[0][d];
  for(n=1;(n<atoms_2.nr);n++)
    for(d=0;(d<DIM);d++) {
      xmin[d]=min(xmin[d],x_2[n][d]);
    }
  for(n=0;(n<atoms_2.nr);n++) 
    for(d=0;(d<DIM);d++) 
      x_2[n][d]-=xmin[d];
  
#ifdef DEBUG
  print_stat(x_2,atoms_2.nr,box_2);
#endif
  /* Sort the solvent mixture, not the protein... */
  m_sort(&atoms_2,x_2,v_2,r_2,atoms_2.atom[atoms_2.nr].resnr,
	 atoms_2.nres-1);
  
  /*add the two configurations*/
  onr_1=atoms_1->nr;
  onres_1=atoms_1->nres;
  srenew(atoms_1->resname,(atoms_1->nres+atoms_2.nres));
  srenew(atoms_1->atomname,(atoms_1->nr+atoms_2.nr));
  srenew(atoms_1->atom,(atoms_1->nr+atoms_2.nr));
  srenew(*x_1,(atoms_1->nr+atoms_2.nr));
  srenew(*v_1,(atoms_1->nr+atoms_2.nr));
  srenew(*r_1,(atoms_1->nr+atoms_2.nr));
  
  add_conf(atoms_1,*x_1,*v_1,*r_1,ntb,box_1,
	   &atoms_2,x_2,v_2,r_2,TRUE,maxmol);
  *atoms_added=atoms_1->nr-onr_1;
  *residues_added=atoms_1->nres-onres_1;
  
  fprintf(stderr,"Generated solvent containing %d atoms in %d residues\n",
	  *atoms_added,*residues_added);
}

char *read_prot(char *confin_1,int ntb,bool bRotate,
		t_atoms *atoms_1,
		rvec **x_1,rvec **v_1,real **r_1,
		matrix box_1,rvec angle,
		t_vdw **vdw,int *max_vdw,real r_distance)
{
  char *title;
  
  snew(title,STRLEN);
  get_stx_coordnum(confin_1,&(atoms_1->nr));
  
  /*allocate memory for atom coordinates of configuration 1*/
  snew(*x_1,atoms_1->nr);
  snew(*v_1,atoms_1->nr);
  snew(*r_1,atoms_1->nr);
  snew(atoms_1->resname,atoms_1->nr);
  snew(atoms_1->atom,atoms_1->nr);
  snew(atoms_1->atomname,atoms_1->nr);
  
  /*read residue number, residue names, atomnames, coordinates etc.*/
  fprintf(stderr,"Reading solute configuration \n");
  read_stx_conf(confin_1,title,atoms_1,*x_1,*v_1,box_1);
  fprintf(stderr,"%s\nContaining %d atoms in %d residues\n",
	  title,atoms_1->nr,atoms_1->nres);
  /*srenew(atoms_1->resname,atoms_1->nres);*/
    
  /*initialise van der waals arrays of configuration 1*/
  mk_vdw(atoms_1,*r_1,vdw,max_vdw,r_distance);
  
  /*orient configuration along z-axis*/
  if (bRotate)
    orient(atoms_1->nr,*x_1,*v_1,angle,box_1); 
  
  return title;
}

int main(int argc,char *argv[])
{
  static char *desc[] = {
    "Genbox can do one of 3 things:[PAR]",
    
    "1) Generate a box of solvent. Specify -cs and -box.[PAR]",
    
    "2) Solvate a solute configuration, eg. a protein, in a bath of solvent ",
    "molecules. Specify [TT]-cp[tt] (solute) and [TT]-cs[tt] (solvent). ",
    "The box specified in the solute coordinate file ([TT]-cp[tt]) is used.",
    "The box can be modified with the program [TT]editconf[tt].",
    "Solvent molecules are removed from the box where the ",
    "distance between any atom of the solute molecule(s) and any atom of ",
    "the solvent molecule is less than the sum of the VanderWaals radii of ",
    "both atoms. A database of VanderWaals radii is read by the program ",
    "(usually this is [TT]vdw.gmx[tt]), atoms not in the database are ",
    "assigned a default distance [TT]-vdw[tt], default 0.105 nm.[PAR]",
    
    "3) Insert a number ([TT]-nmol[tt]) of extra molecules ([TT]-ci[tt]) ",
    "at random positions.",
    "The program iterates until [TT]nmol[tt] molecules",
    "have been inserted in the box. To test whether an insertion is ",
    "successful the same VanderWaals criterium is used as for removal of ",
    "solvent molecules. When no appropriately ",
    "sized holes (holes that can hold an extra molecule) are available the ",
    "program does not terminate, but searches forever. To avoid this problem ",
    "the genbox program may be used several times in a row with a smaller ",
    "number of molecules to be inserted. Alternatively, you can add the ",
    "extra molecules to the solute first, and then in a second run of ",
    "genbox solvate it all.",
    
    "The default solvent is Simple Point Charge water (SPC). The coordinates ",
    "for this are read from [TT]$GMXLIB/spc216.gro[tt]. Optionally a maximum ",
    "number of solvent molecules ([TT]-maxsol[tt]) can be specified. Other",
    "solvents are also supported, as well as mixed solvents. The",
    "only restriction to solvent types is that a solvent molecule consists",
    "of exactly one residue. The residue information in the coordinate",
    "files is used, and should therefore be more or less consistent.",
    "In practice this means that two subsequent solvent molecules in the ",
    "solvent coordinate file should have different residue number.",
    "The box of solute is built by stacking the coordinates read from",
    "the coordinate file. This means that these coordinates should be ",
    "equlibrated in periodic boundary conditions to ensure a good",
    "alignment of molecules on the stacking interfaces.[PAR]",
    
    "The program can optionally rotate the solute molecule to align the",
    "longest molecule axis along a box edge. This way the amount of solvent",
    "molecules necessary is reduced.",
    "It should be kept in mind that this only works for",
    "short simulations, as eg. an alpha-helical peptide in solution can ",
    "rotate over 90 degrees, within 500 ps. In general it is therefore ",
    "better to make a more or less cubic box.[PAR]",
    
    "Finally, genbox will optionally remove lines from your topology file in ",
    "which a number of solvent molecules is already added, and adds a ",
    "line with the total number of solvent molecules in your coordinate file."
  };

  static char *bugs[] = {
    
    "The program does not take periodic boundary conditions in the initial "
    "configuration into account, therefore the solute configuration should "
    "only contain whole molecules. This can be especially tricky when using "
    "a coordinate file from a protein crystal.",
    
    "Because of the way the VanderWaals criterium is checked, genbox may take "
    "quite long (minutes on a workstation) to complete.",
    
    "For very large boxes, genbox may rapidly run out of memory."
  };
  
  /* parameter data */
  bool bSol,bProt;
  char *confin_1,*confin_2,*confin_3,*confout,*topinout;
  int  bInsert;
  real *r_1;
  char *title_3;
  
  /* protein configuration data */
  char    *title;
  t_atoms atoms_1;
  rvec    *x_1,*v_1;
  matrix  box_1;
  
  /* other data types */
  int  atoms_added,residues_added;
  
  /* van der waals data */
  int    max_vdw;
  t_vdw  *vdw;
  
  /*angle variables*/
  rvec angle;
  t_filenm fnm[] = {
    { efSTX, "-cp", "protein", ffOPTRD },
    { efSTX, "-cs", "spc216",  ffLIBOPTRD},
    { efSTX, "-ci", "insert",  ffOPTRD},
    { efVDW, "-w",  NULL,      ffLIBRD},
    { efSTO, NULL,  NULL,      ffWRITE},
    { efTOP, NULL,  NULL,      ffOPTRW},
  };
#define NFILE asize(fnm)
  
  static int nmol_3=0,maxsol=0,seed=1997,ntb=0;
  static bool bRotate=FALSE;
  static real r_distance=0.105;
  static rvec new_box={0.0,0.0,0.0};
  t_pargs pa[] = {
    { "-box",   FALSE,etRVEC,&new_box,   "box size" },
    { "-nmol",  FALSE,etINT ,&nmol_3,    "no of extra molecules to insert" },
    { "-maxsol",FALSE,etINT ,&maxsol,    "max no of solvent molecules to add"},
    { "-rot",   FALSE,etBOOL,&bRotate,   "rotate solute to fit box best"},
    { "-seed",  FALSE,etINT ,&seed,      "random generator seed"},
    { "-vdwd",  FALSE,etREAL,&r_distance,"default vdwaals distance"},
    { "-boxtype",FALSE,etINT,&ntb, "HIDDENbox type 0=rectangular; "
      "1=truncated octahedron (only rectangular boxes are fully implemented)"}
  };

  CopyRight(stderr,argv[0]);
  parse_common_args(&argc,argv,0,TRUE,NFILE,fnm,asize(pa),pa,
		    asize(desc),desc,asize(bugs),bugs);
  
  /* Copy filenames... */
  confin_1=opt2fn("-cp",NFILE,fnm);
  confin_2=opt2fn("-cs",NFILE,fnm);
  confin_3=opt2fn("-ci",NFILE,fnm);
  confout =ftp2fn(efSTO,NFILE,fnm);
  topinout=ftp2fn(efTOP,NFILE,fnm);
  
  bInsert = opt2bSet("-ci",NFILE,fnm);
  bSol = opt2bSet("-cs",NFILE,fnm);
  bProt= opt2bSet("-cp",NFILE,fnm);
  if (!bProt && !opt2parg_bSet("-box",asize(pa),pa)) {
    fprintf(stderr,"You must supply a solute (-cp) or a box size (-box)\n\n");
    exit(0);
  }
  /* read van der waals distances for all the existing atoms*/
  vdw=NULL;
  max_vdw=read_vdw(ftp2fn(efVDW,NFILE,fnm),&vdw);

  if (bProt) {
    /*generate a solute configuration*/
    title=read_prot(confin_1,ntb,bRotate,&atoms_1,&x_1,&v_1,&r_1,box_1,angle,
		    &vdw,&max_vdw,r_distance);
    if (atoms_1.nr == 0) {
      fprintf(stderr,"No protein in %s, check your input\n",confin_1);
      exit(1);
    }
  }
  else {
    /* in case there is no a solute,
     * the box size is set to new_box
     */
    clear_mat(box_1);
    box_1[XX][XX]=new_box[XX];
    box_1[YY][YY]=new_box[YY];
    box_1[ZZ][ZZ]=new_box[ZZ];
    atoms_1.nr=0;
    atoms_1.nres=0;
    bRotate=FALSE;
    atoms_1.resname=NULL;
    atoms_1.atomname=NULL;
    atoms_1.atom=NULL;
    x_1=NULL;
    v_1=NULL;
    r_1=NULL;
  }
  init_pbc(box_1,ntb);
  
  /*add nmol_3 molecules of atoms_3 in random orientation at random place*/
  if (bInsert && (nmol_3 > 0))
    title_3=add_mol3(confin_3,nmol_3,seed,ntb,
		     &atoms_1,&x_1,&v_1,&r_1,box_1,
		     &vdw,&max_vdw,r_distance);
  else
    title_3=strdup("Generated by genbox");
  
  if (bSol) 
    add_solv(confin_2,ntb,&atoms_1,&x_1,&v_1,&r_1,box_1,
	     &vdw,&max_vdw,r_distance,&atoms_added,&residues_added,maxsol);
  
  if (debug)
    write_vdw("outradii.vdw",vdw,max_vdw);

  /*write new configuration 1 to file confout*/
  fprintf(stderr,"Writing generated configuration to disk\n");
 
  if (bProt) {
    write_sto_conf(confout,title,&atoms_1,x_1,v_1,box_1);
    /*print box sizes and box type to stderr*/
    fprintf(stderr,"%s\n",title);  
    fprintf(stderr,"Generated a %s box sizes: %f %f %f\n",
	    eboxtype_names[ntb],box_1[XX][XX],box_1[YY][YY],box_1[ZZ][ZZ]);
  }
  else 
    write_sto_conf(confout,title_3,&atoms_1,x_1,v_1,box_1);

  if (debug)
    write_vdw("out.vdw",vdw,max_vdw);
  
  /*print rotation data*/
  if (bRotate)
    fprintf(stderr,"Configuration is rotated \nalfa=%f beta=%f gamma=%f\n",
	    angle[0],angle[1],angle[2]);
  
  /* print size of generated configuration */
  fprintf(stderr,"\nOutput configuration contains %d atoms in %d residues\n",
	  atoms_1.nr,atoms_1.nres);
  {
    int  i,nsol=0;
    real vol;
  
    for(i=0; (i<atoms_1.nres); i++) {
      /* calculate number of SOLvent molecules */
      if (strcmp(*atoms_1.resname[i],"SOL")==0)
	nsol++;
    }
    
    vol=det(box_1);
    
    fprintf(stderr,"Volume                 :  %10g (nm^3)\n",vol);
    fprintf(stderr,"Number of SOL molecules:  %5d   \n\n",nsol);
    
    /* open topology file and append sol molecules */
    if ( ftp2bSet(efTOP,NFILE,fnm) ) {
      FILE *fpin,*fpout;
      char buf[STRLEN],*temp;
      int  line;
      
      fprintf(stderr,"Processing topology\n");
      fpin = ffopen(topinout,"r");
      fpout= ffopen("temp.top","w");
      line=0;
      while (fgets(buf, STRLEN, fpin)) {
	line++;
	if (strncmp("SOL       ",buf,10)!=0)
	  fprintf(fpout,"%s",buf);
	else {
	  if (temp=strchr(buf,'\n'))
	    temp[0]='\0';
	  fprintf(stdout,"Removing line #%d '%s' from topology file (%s)\n",
		  line,buf,topinout);
	}  
      }
      if ( nsol > 0 ) {
	fprintf(stdout,"Adding line for %d solute molecules to "
		"topology file (%s)\n",nsol,topinout);
	fprintf(fpout,"SOL       %5d\n",nsol);
      } else
	fprintf(stdout,"No SOL molecules, not adding anything to "
		"topology file (%s)\n",topinout);
      fclose(fpout);
      fclose(fpin);
      rename("temp.top",topinout);
    }
  }
  thanx(stdout);
  
  return 0;
}
