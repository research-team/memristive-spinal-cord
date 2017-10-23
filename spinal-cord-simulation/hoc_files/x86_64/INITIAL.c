/* Created by Language version: 6.2.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__initial
#define _nrn_initial _nrn_initial__initial
#define nrn_cur _nrn_cur__initial
#define _nrn_current _nrn_current__initial
#define nrn_jacob _nrn_jacob__initial
#define nrn_state _nrn_state__initial
#define _net_receive _net_receive__initial 
#define evaluate_fct evaluate_fct__initial 
#define states states__initial 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gnabar _p[0]
#define gl _p[1]
#define gkrect _p[2]
#define gnap _p[3]
#define ena _p[4]
#define ek _p[5]
#define el _p[6]
#define ina _p[7]
#define il _p[8]
#define ikrect _p[9]
#define inap _p[10]
#define m_inf _p[11]
#define h_inf _p[12]
#define n_inf _p[13]
#define p_inf _p[14]
#define tau_m _p[15]
#define tau_h _p[16]
#define tau_p _p[17]
#define tau_n _p[18]
#define p _p[19]
#define m _p[20]
#define h _p[21]
#define n _p[22]
#define Dp _p[23]
#define Dm _p[24]
#define Dh _p[25]
#define Dn _p[26]
#define v _p[27]
#define _g _p[28]
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_Exp(void);
 static void _hoc_alpham(void);
 static void _hoc_betam(void);
 static void _hoc_evaluate_fct(void);
 static void _hoc_vtrap2(void);
 static void _hoc_vtrap1(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_initial", _hoc_setdata,
 "Exp_initial", _hoc_Exp,
 "alpham_initial", _hoc_alpham,
 "betam_initial", _hoc_betam,
 "evaluate_fct_initial", _hoc_evaluate_fct,
 "vtrap2_initial", _hoc_vtrap2,
 "vtrap1_initial", _hoc_vtrap1,
 0, 0
};
#define Exp Exp_initial
#define alpham alpham_initial
#define betam betam_initial
#define vtrap2 vtrap2_initial
#define vtrap1 vtrap1_initial
 extern double Exp( _threadargsprotocomma_ double );
 extern double alpham( _threadargsprotocomma_ double );
 extern double betam( _threadargsprotocomma_ double );
 extern double vtrap2( _threadargsprotocomma_ double );
 extern double vtrap1( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define ampC ampC_initial
 double ampC = 5;
#define ampB ampB_initial
 double ampB = 28.4;
#define ampA ampA_initial
 double ampA = 0.0353;
#define amC amC_initial
 double amC = 5;
#define amB amB_initial
 double amB = 60;
#define amA amA_initial
 double amA = 0.4;
#define bmpC bmpC_initial
 double bmpC = 5;
#define bmpB bmpB_initial
 double bmpB = 32.7;
#define bmpA bmpA_initial
 double bmpA = 0.000883;
#define bmC bmC_initial
 double bmC = 5;
#define bmB bmB_initial
 double bmB = 40;
#define bmA bmA_initial
 double bmA = 0.4;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gnabar_initial", "mho/cm2",
 "gl_initial", "mho/cm2",
 "gkrect_initial", "mho/cm2",
 "gnap_initial", "mho/gm2",
 "ena_initial", "mV",
 "ek_initial", "mV",
 "el_initial", "mV",
 "ina_initial", "mA/cm2",
 "il_initial", "mA/cm2",
 "ikrect_initial", "mA/cm2",
 "inap_initial", "mA/cm2",
 0,0
};
 static double delta_t = 1;
 static double h0 = 0;
 static double m0 = 0;
 static double n0 = 0;
 static double p0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "amA_initial", &amA_initial,
 "amB_initial", &amB_initial,
 "amC_initial", &amC_initial,
 "bmA_initial", &bmA_initial,
 "bmB_initial", &bmB_initial,
 "bmC_initial", &bmC_initial,
 "ampA_initial", &ampA_initial,
 "ampB_initial", &ampB_initial,
 "ampC_initial", &ampC_initial,
 "bmpA_initial", &bmpA_initial,
 "bmpB_initial", &bmpB_initial,
 "bmpC_initial", &bmpC_initial,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[0]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "6.2.0",
"initial",
 "gnabar_initial",
 "gl_initial",
 "gkrect_initial",
 "gnap_initial",
 "ena_initial",
 "ek_initial",
 "el_initial",
 0,
 "ina_initial",
 "il_initial",
 "ikrect_initial",
 "inap_initial",
 "m_inf_initial",
 "h_inf_initial",
 "n_inf_initial",
 "p_inf_initial",
 "tau_m_initial",
 "tau_h_initial",
 "tau_p_initial",
 "tau_n_initial",
 0,
 "p_initial",
 "m_initial",
 "h_initial",
 "n_initial",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 29, _prop);
 	/*initialize range parameters*/
 	gnabar = 0.5;
 	gl = 0.01;
 	gkrect = 0.1;
 	gnap = 0.01;
 	ena = 50;
 	ek = -80;
 	el = -70;
 	_prop->param = _p;
 	_prop->param_size = 29;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 1, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _INITIAL_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
  hoc_register_prop_size(_mechtype, 29, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 initial /home/neucogar/Desktop/spinal-cord/hoc_files/x86_64/INITIAL.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "Motor Neuron initial segment";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int evaluate_fct(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[4], _dlist1[4];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset = 0; {
   evaluate_fct ( _threadargscomma_ v ) ;
   Dm = ( m_inf - m ) / tau_m ;
   Dh = ( h_inf - h ) / tau_h ;
   Dp = ( p_inf - p ) / tau_p ;
   Dn = ( n_inf - n ) / tau_n ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
 evaluate_fct ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_m )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_h )) ;
 Dp = Dp  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_p )) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_n )) ;
 return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) { {
   evaluate_fct ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_m)))*(- ( ( ( m_inf ) ) / tau_m ) / ( ( ( ( - 1.0 ) ) ) / tau_m ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_h)))*(- ( ( ( h_inf ) ) / tau_h ) / ( ( ( ( - 1.0 ) ) ) / tau_h ) - h) ;
    p = p + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_p)))*(- ( ( ( p_inf ) ) / tau_p ) / ( ( ( ( - 1.0 ) ) ) / tau_p ) - p) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_n)))*(- ( ( ( n_inf ) ) / tau_n ) / ( ( ( ( - 1.0 ) ) ) / tau_n ) - n) ;
   }
  return 0;
}
 
static int  evaluate_fct ( _threadargsprotocomma_ double _lv ) {
   double _la , _lb , _lv2 ;
 _la = alpham ( _threadargscomma_ _lv ) ;
   _lb = betam ( _threadargscomma_ _lv ) ;
   tau_m = 1.0 / ( _la + _lb ) ;
   m_inf = _la / ( _la + _lb ) ;
   tau_h = 30.0 / ( Exp ( _threadargscomma_ ( _lv + 60.0 ) / 15.0 ) + Exp ( _threadargscomma_ - ( _lv + 60.0 ) / 16.0 ) ) ;
   h_inf = 1.0 / ( 1.0 + Exp ( _threadargscomma_ ( _lv + 65.0 ) / 7.0 ) ) ;
   _la = vtrap1 ( _threadargscomma_ _lv ) ;
   _lb = vtrap2 ( _threadargscomma_ _lv ) ;
   tau_p = 1.0 / ( _la + _lb ) ;
   p_inf = _la / ( _la + _lb ) ;
   tau_n = 5.0 / ( Exp ( _threadargscomma_ ( _lv + 50.0 ) / 40.0 ) + Exp ( _threadargscomma_ - ( _lv + 50.0 ) / 50.0 ) ) ;
   n_inf = 1.0 / ( 1.0 + Exp ( _threadargscomma_ - ( _lv + 38.0 ) / 15.0 ) ) ;
    return 0; }
 
static void _hoc_evaluate_fct(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 evaluate_fct ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double alpham ( _threadargsprotocomma_ double _lx ) {
   double _lalpham;
 if ( fabs ( ( _lx + amB ) / amC ) < 1e-6 ) {
     _lalpham = amA * amC ;
     }
   else {
     _lalpham = ( amA * ( _lx + amB ) ) / ( 1.0 - Exp ( _threadargscomma_ - ( _lx + amB ) / amC ) ) ;
     }
   
return _lalpham;
 }
 
static void _hoc_alpham(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alpham ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double betam ( _threadargsprotocomma_ double _lx ) {
   double _lbetam;
 if ( fabs ( ( _lx + bmB ) / bmC ) < 1e-6 ) {
     _lbetam = - bmA * bmC ;
     }
   else {
     _lbetam = ( bmA * ( - ( _lx + bmB ) ) ) / ( 1.0 - Exp ( _threadargscomma_ ( _lx + bmB ) / bmC ) ) ;
     }
   
return _lbetam;
 }
 
static void _hoc_betam(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betam ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double vtrap1 ( _threadargsprotocomma_ double _lx ) {
   double _lvtrap1;
 if ( fabs ( ( _lx + ampB ) / ampC ) < 1e-6 ) {
     _lvtrap1 = ampA * ampC ;
     }
   else {
     _lvtrap1 = ( ampA * ( _lx + ampB ) ) / ( 1.0 - Exp ( _threadargscomma_ - ( _lx + ampB ) / ampC ) ) ;
     }
   
return _lvtrap1;
 }
 
static void _hoc_vtrap1(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  vtrap1 ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double vtrap2 ( _threadargsprotocomma_ double _lx ) {
   double _lvtrap2;
 if ( fabs ( ( _lx + bmpB ) / bmpC ) < 1e-6 ) {
     _lvtrap2 = - bmpA * bmpC ;
     }
   else {
     _lvtrap2 = ( bmpA * ( - ( _lx + bmpB ) ) ) / ( 1.0 - Exp ( _threadargscomma_ ( _lx + bmpB ) / bmpC ) ) ;
     }
   
return _lvtrap2;
 }
 
static void _hoc_vtrap2(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  vtrap2 ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double Exp ( _threadargsprotocomma_ double _lx ) {
   double _lExp;
 if ( _lx < - 100.0 ) {
     _lExp = 0.0 ;
     }
   else {
     _lExp = exp ( _lx ) ;
     }
   
return _lExp;
 }
 
static void _hoc_Exp(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  Exp ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 4;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 4; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
  n = n0;
  p = p0;
 {
   evaluate_fct ( _threadargscomma_ v ) ;
   m = m_inf ;
   h = h_inf ;
   p = p_inf ;
   n = n_inf ;
   }
 
}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ina = gnabar * m * m * m * h * ( v - ena ) ;
   ikrect = gkrect * n * n * n * n * ( v - ek ) ;
   inap = gnap * p * p * p * ( v - ena ) ;
   il = gl * ( v - el ) ;
   }
 _current += ina;
 _current += ikrect;
 _current += inap;
 _current += il;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 {   states(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(m) - _p;  _dlist1[0] = &(Dm) - _p;
 _slist1[1] = &(h) - _p;  _dlist1[1] = &(Dh) - _p;
 _slist1[2] = &(p) - _p;  _dlist1[2] = &(Dp) - _p;
 _slist1[3] = &(n) - _p;  _dlist1[3] = &(Dn) - _p;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif
