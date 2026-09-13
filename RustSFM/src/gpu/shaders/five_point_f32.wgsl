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

// Symmetric Jacobi eigensolver for A^T A. The four smallest eigenvectors
// are the right nullspace of the fixed 5x9 constraint matrix.
@group(0) @binding(0) var<storage, read> constraints: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn nullspace(@builtin(workgroup_id) id: vec3<u32>) {
  let n = id.x; let ib = n * 45u; let ob = n * 40u;
  // Record: basis[36], status, sweeps, relative off-diagonal, numerical rank.
  var scale=0.0;
  for(var i=0u;i<45u;i++) { scale=max(scale,abs(constraints[ib+i])); }
  if(scale==0.0) { output[ob+36u]=2.0; return; }
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
    output[ob+37u]=f32(sweep+1u); output[ob+38u]=off/trace;
    if(off<=2e-8*trace) { converged=true; break; }
  }
  if(!converged) { output[ob+36u]=1.0; return; }
  var rank=0u;
  for(var i=0u;i<9u;i++) { if(s[i*9u+i]>1e-6*trace) { rank++; }}
  output[ob+39u]=f32(rank);
  if(rank!=5u) { output[ob+36u]=2.0; return; }
  // Fixed-size selection sort eigenpairs ascending; columns of v are vectors.
  for(var k:u32=0u;k<4u;k++){var best=k;for(var j:u32=k+1u;j<9u;j++){if(s[j*9u+j]<s[best*9u+best]){best=j;}}
    for(var r:u32=0u;r<9u;r++){let x=v[r*9u+k];v[r*9u+k]=v[r*9u+best];v[r*9u+best]=x;} let x=s[k*9u+k];s[k*9u+k]=s[best*9u+best];s[best*9u+best]=x;
    for(var r:u32=0u;r<9u;r++){output[ob+k*9u+r]=v[r*9u+k];}
  }
  // Independently verify A*N and N^T*N before reporting success.
  for(var k=0u;k<4u;k++) {
    for(var j=0u;j<4u;j++) {
      var dot=0.0;
      for(var r=0u;r<9u;r++) { dot+=v[r*9u+k]*v[r*9u+j]; }
      if(!(abs(dot-select(0.0,1.0,k==j))<2e-4)) { output[ob+36u]=5.0; return; }
    }
    for(var r=0u;r<5u;r++) {
      var residual=0.0;
      for(var j=0u;j<9u;j++) { residual+=(constraints[ib+r*9u+j]/scale)*v[j*9u+k]; }
      if(!(abs(residual)<5e-4*sqrt(trace))) { output[ob+36u]=5.0; return; }
    }
  }
  output[ob+36u]=0.0;
}
