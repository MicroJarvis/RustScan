// Appended after five_point_generated.wgsl. One invocation per sample.
@group(0) @binding(0) var<storage, read> bases: array<f32>;
// Record: a[200], solve[100], b[39], coeffs[11], status, minimum relative pivot.
@group(0) @binding(1) var<storage, read_write> algebra_output: array<f32>;

fn finite(x: f32) -> bool { return x == x && abs(x) <= 3.402823e38; }

@compute @workgroup_size(1)
fn algebra(@builtin(workgroup_id) id: vec3<u32>) {
  let ob = id.x * 352u;
  var a: array<f32,200>;
  for(var i=0u;i<200u;i++) { a[i]=algebra_output[ob+i]; }
  var scale = 0.0;
  for(var i=0u; i<200u; i++) {
    algebra_output[ob+i] = a[i];
    if (!finite(a[i])) { algebra_output[ob+350u]=4.0; return; }
    if(i<100u) { scale=max(scale,abs(a[i])); }
  }
  var min_pivot=1.0;
  // Partial row pivoting on the 10x20 augmented matrix; no transpose or sign flip.
  for(var k=0u; k<10u; k++) {
    var pivot=k;
    for(var r=k+1u; r<10u; r++) {
      if(abs(a[k*10u+r])>abs(a[k*10u+pivot])) { pivot=r; }
    }
    let magnitude=abs(a[k*10u+pivot]);
    if(scale==0.0 || magnitude <= 1e-7*scale) {
      algebra_output[ob+350u]=3.0; algebra_output[ob+351u]=0.0; return;
    }
    min_pivot=min(min_pivot,magnitude/scale);
    for(var c=0u; c<20u; c++) {
      let t=a[c*10u+k]; a[c*10u+k]=a[c*10u+pivot]; a[c*10u+pivot]=t;
    }
    let diagonal=a[k*10u+k];
    for(var r=k+1u; r<10u; r++) {
      let factor=a[k*10u+r]/diagonal;
      a[k*10u+r]=0.0;
      for(var c=k+1u; c<20u; c++) { a[c*10u+r]-=factor*a[c*10u+k]; }
    }
  }
  var solved: array<f32,100>;
  for(var c=0u; c<10u; c++) {
    for(var reverse=0u; reverse<10u; reverse++) {
      let r=9u-reverse;
      var x=a[(c+10u)*10u+r];
      for(var j=r+1u; j<10u; j++) { x-=a[j*10u+r]*solved[c*10u+j]; }
      solved[c*10u+r]=x/a[r*10u+r];
    }
  }
  var b: array<f32,39>;
  for(var i=0u; i<3u; i++) {
    for(var k=0u; k<3u; k++) {
      b[i*13u+1u+k]=solved[k*10u+i*2u+4u];
      b[i*13u+5u+k]=solved[(3u+k)*10u+i*2u+4u];
    }
    for(var k=0u; k<4u; k++) { b[i*13u+9u+k]=solved[(6u+k)*10u+i*2u+4u]; }
    for(var k=0u; k<3u; k++) {
      b[i*13u+k]-=solved[k*10u+i*2u+5u];
      b[i*13u+4u+k]-=solved[(3u+k)*10u+i*2u+5u];
    }
    for(var k=0u; k<4u; k++) { b[i*13u+8u+k]-=solved[(6u+k)*10u+i*2u+5u]; }
  }
  var status=0.0;
  for(var i=0u; i<100u; i++) {
    algebra_output[ob+200u+i]=solved[i]; if(!finite(solved[i])) { status=4.0; }
  }
  for(var i=0u; i<39u; i++) {
    algebra_output[ob+300u+i]=b[i]; if(!finite(b[i])) { status=4.0; }
  }
  algebra_output[ob+350u]=status; algebra_output[ob+351u]=min_pivot;
}

@compute @workgroup_size(1)
fn validate_polynomial(@builtin(workgroup_id) id: vec3<u32>) {
  let ob=id.x*352u;
  if(algebra_output[ob+350u]!=0.0) { return; }
  var coeffs: array<f32,11>;
  for(var i=0u;i<11u;i++) { coeffs[i]=algebra_output[ob+339u+i]; }
  var status=0.0;
  var coeff_scale=0.0;
  for(var i=0u; i<11u; i++) {
    algebra_output[ob+339u+i]=coeffs[i];
    coeff_scale=max(coeff_scale,abs(coeffs[i]));
    if(!finite(coeffs[i])) { status=4.0; }
  }
  if(coeff_scale==0.0) { status=3.0; }
  algebra_output[ob+350u]=status;
}
