struct Ray { xyz: vec3<f32>, _pad: f32 };
@group(0) @binding(0) var<storage, read> rays1: array<Ray>;
@group(0) @binding(1) var<storage, read> rays2: array<Ray>;
@group(0) @binding(2) var<storage, read_write> matrix_output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) id: vec3<u32>) {
  let batch = id.x;
  for (var r: u32 = 0u; r < 5u; r++) {
    let i = batch * 5u + r;
    let a = rays1[i].xyz; let b = rays2[i].xyz;
    let base = batch * 45u + r * 9u;
    matrix_output[base+0u] = b.x*a.x; matrix_output[base+1u] = b.x*a.y; matrix_output[base+2u] = b.x*a.z;
    matrix_output[base+3u] = b.y*a.x; matrix_output[base+4u] = b.y*a.y; matrix_output[base+5u] = b.y*a.z;
    matrix_output[base+6u] = b.z*a.x; matrix_output[base+7u] = b.z*a.y; matrix_output[base+8u] = b.z*a.z;
  }
}

// Q2 nullspace rank gate: act on singular values, not on eigenvalues of AᵀA.
//
// RANK CRITERION (written before implementation):
//   Scale A by its max-abs entry. Gate σ from the 5×5 Gram AAᵀ (Jacobi → σ²),
//   not from the 9×9 AᵀA: f32 Jacobi on AᵀA leaves a ~1e-5 noise floor on
//   numerical-null singular values, which overlaps the rescue band
//   σ₅/σ₁ ∈ [1e-5, 1e-2). AAᵀ has only the five true singular values, so
//   rank-deficient samples fall below the floor while mild rank-5 samples
//   stay measurable. Accept iff
//     σ₅/σ₁ > 1e-5
//   (1e-6 still lets f32 AAᵀ noise on true σ₅≈0 samples through; 1e-5 sits
//   at the bottom of the rescue band [1e-5, 1e-2).) Nullspace still from
//   AᵀA eigenvectors for the four smallest eigenvalues (QR of Aᵀ prototyped
//   as对照: A*N≈0 passes, but a synthetic algebra fixture breaks the f64
//   five-point determinant identity ~1e-2). Export σ₅/σ₁ at output[38]; on
//   success rank=5, on failure #{σ_i > 1e-5·σ₁} from the five AAᵀ values.
//
// Record: basis[36], status, sweeps, sigma5_over_sigma1, numerical rank.
// P2: trial count comes from params.count, never from arrayLength.
struct FivePointParams { count: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<storage, read> constraints: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: FivePointParams;

@compute @workgroup_size(32)
fn nullspace(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.count) { return; }
  let n = id.x; let ib = n * 45u; let ob = n * 40u;
  var scale=0.0;
  for(var i=0u;i<45u;i++) { scale=max(scale,abs(constraints[ib+i])); }
  if(scale==0.0) { output[ob+36u]=2.0; return; }

  // --- Gate: Jacobi on 5×5 AAᵀ → σ₁…σ₅ ---
  var g: array<f32,25>;
  for (var r:u32=0u;r<5u;r++) { for (var c:u32=0u;c<5u;c++) {
    var x:f32=0.0;
    for (var j:u32=0u;j<9u;j++) {
      x += (constraints[ib+r*9u+j]/scale)*(constraints[ib+c*9u+j]/scale);
    }
    g[r*5u+c]=x;
  }}
  var gtrace=0.0;
  for(var i=0u;i<5u;i++) { gtrace+=g[i*5u+i]; }
  var g_ok=false;
  var g_sweeps=0u;
  for (var sweep=0u;sweep<64u;sweep++) {
    for (var p=0u;p<4u;p++) { for (var q=p+1u;q<5u;q++) {
      let apq=g[p*5u+q]; if(abs(apq)<=1e-8*gtrace) { continue; }
      let tau=(g[q*5u+q]-g[p*5u+p])/(2.0*apq);
      let t=select(-1.0,1.0,tau>=0.0)/(abs(tau)+sqrt(1.0+tau*tau));
      let c=inverseSqrt(1.0+t*t); let z=t*c;
      let app=g[p*5u+p]; let aqq=g[q*5u+q];
      for(var k=0u;k<5u;k++) {
        if(k!=p && k!=q) {
          let x=g[k*5u+p]; let y=g[k*5u+q];
          g[k*5u+p]=c*x-z*y; g[p*5u+k]=g[k*5u+p];
          g[k*5u+q]=z*x+c*y; g[q*5u+k]=g[k*5u+q];
        }
      }
      g[p*5u+p]=app-t*apq; g[q*5u+q]=aqq+t*apq;
      g[p*5u+q]=0.0; g[q*5u+p]=0.0;
    }}
    var off=0.0;
    for(var p=0u;p<4u;p++) { for(var q=p+1u;q<5u;q++) { off=max(off,abs(g[p*5u+q])); }}
    g_sweeps=sweep+1u;
    if(off<=2e-8*gtrace) { g_ok=true; break; }
  }
  if(!g_ok) { output[ob+36u]=1.0; output[ob+37u]=f32(g_sweeps); return; }
  for(var k=0u;k<5u;k++){var best=k;for(var j=k+1u;j<5u;j++){if(g[j*5u+j]>g[best*5u+best]){best=j;}}
    let x=g[k*5u+k];g[k*5u+k]=g[best*5u+best];g[best*5u+best]=x;
  }
  let sig1=sqrt(max(g[0],0.0));
  let sig5=sqrt(max(g[4u*5u+4u],0.0));
  let ratio=select(0.0, sig5/sig1, sig1>0.0);
  output[ob+38u]=ratio;
  var above=0u;
  for(var i=0u;i<5u;i++) {
    if(sqrt(max(g[i*5u+i],0.0))>1e-5*sig1) { above++; }
  }
  if(!(ratio>1e-5)) {
    output[ob+39u]=f32(above);
    output[ob+37u]=f32(g_sweeps);
    output[ob+36u]=2.0;
    return;
  }

  // --- Basis: Jacobi on 9×9 AᵀA (unchanged path; algebra-compatible) ---
  var s: array<f32,81>; var v: array<f32,81>;
  for (var i:u32=0u;i<9u;i++) { for (var j:u32=0u;j<9u;j++) {
    var x:f32=0.0; for (var r:u32=0u;r<5u;r++){x += (constraints[ib+r*9u+i]/scale)*(constraints[ib+r*9u+j]/scale);}
    s[i*9u+j]=x; v[i*9u+j]=select(0.0,1.0,i==j);
  }}
  var trace=0.0;
  for(var i=0u;i<9u;i++) { trace+=s[i*9u+i]; }
  var converged=false;
  for (var sweep=0u;sweep<64u;sweep++) {
    for (var p=0u;p<8u;p++) { for (var q=p+1u;q<9u;q++) {
      let apq=s[p*9u+q]; if(abs(apq)<=1e-8*trace) { continue; }
      let tau=(s[q*9u+q]-s[p*9u+p])/(2.0*apq);
      let t=select(-1.0,1.0,tau>=0.0)/(abs(tau)+sqrt(1.0+tau*tau));
      let c=inverseSqrt(1.0+t*t); let z=t*c;
      let app=s[p*9u+p]; let aqq=s[q*9u+q];
      for(var k=0u;k<9u;k++) {
        if(k!=p && k!=q) {
          let x=s[k*9u+p]; let y=s[k*9u+q];
          s[k*9u+p]=c*x-z*y; s[p*9u+k]=s[k*9u+p];
          s[k*9u+q]=z*x+c*y; s[q*9u+k]=s[k*9u+q];
        }
        let x=v[k*9u+p]; let y=v[k*9u+q];
        v[k*9u+p]=c*x-z*y; v[k*9u+q]=z*x+c*y;
      }
      s[p*9u+p]=app-t*apq; s[q*9u+q]=aqq+t*apq;
      s[p*9u+q]=0.0; s[q*9u+p]=0.0;
    }}
    var off=0.0;
    for(var p=0u;p<8u;p++) { for(var q=p+1u;q<9u;q++) { off=max(off,abs(s[p*9u+q])); }}
    output[ob+37u]=f32(g_sweeps+sweep+1u);
    if(off<=2e-8*trace) { converged=true; break; }
  }
  if(!converged) { output[ob+36u]=1.0; return; }
  for(var k=0u;k<9u;k++){var best=k;for(var j=k+1u;j<9u;j++){if(s[j*9u+j]>s[best*9u+best]){best=j;}}
    for(var r=0u;r<9u;r++){let x=v[r*9u+k];v[r*9u+k]=v[r*9u+best];v[r*9u+best]=x;}
    let x=s[k*9u+k];s[k*9u+k]=s[best*9u+best];s[best*9u+best]=x;
  }
  output[ob+39u]=5.0;
  for(var k=0u;k<4u;k++){
    let src=5u+k;
    for(var r=0u;r<9u;r++){output[ob+k*9u+r]=v[r*9u+src];}
  }
  for(var k=0u;k<4u;k++) {
    for(var j=0u;j<4u;j++) {
      var dot=0.0;
      for(var r=0u;r<9u;r++) { dot+=output[ob+k*9u+r]*output[ob+j*9u+r]; }
      if(!(abs(dot-select(0.0,1.0,k==j))<2e-4)) { output[ob+36u]=5.0; return; }
    }
    for(var r=0u;r<5u;r++) {
      var residual=0.0;
      for(var j=0u;j<9u;j++) { residual+=(constraints[ib+r*9u+j]/scale)*output[ob+k*9u+j]; }
      if(!(abs(residual)<5e-4*sqrt(trace))) { output[ob+36u]=5.0; return; }
    }
  }
  output[ob+36u]=0.0;
}
