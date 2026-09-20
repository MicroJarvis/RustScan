// Full GPU experiment. Records: 16 header floats + 10 slots of 16 floats.
// Header: upstream, models, degree, iterations, unconverged, complex, duplicate roots,
// recovery rejects, duplicate models, dropped leading, max root backward error,
// rank, Jacobi sweeps, algebra status, max constraint residual, root failure.
// Slot: status, root re/im, backward error, E[9] row-major, B null residual,
// A*E residual, essential cubic residual. Status definitions are in Rust.
// Slot status 3 = real-axis residual gate, 7 = Aberth done[] not set, 8 = polish
// residual; the header unconverged counter still sums all three.
// P2: roots bounds with params.count; recover still uses host dispatch count.
struct FivePointParams { count: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<storage, read> constraints_full: array<f32>;
@group(0) @binding(1) var<storage, read> diagnostics: array<f32>;
@group(0) @binding(2) var<storage, read> algebra_full: array<f32>;
@group(0) @binding(3) var<storage, read> basis_full: array<f32>;
@group(0) @binding(4) var<storage, read_write> results: array<f32>;
@group(0) @binding(5) var<uniform> params: FivePointParams;

fn finite_r(x:f32)->bool { return x==x && abs(x)<=3.402823e38; }
fn finite_v(x:vec2<f32>)->bool { return finite_r(x.x) && finite_r(x.y); }
fn cmul(a:vec2<f32>,b:vec2<f32>)->vec2<f32> { return vec2<f32>(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
fn cdiv(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {
  let scale=max(abs(b.x),abs(b.y));
  if(scale<1e-30) { return vec2<f32>(1e20,1e20); }
  let d=b/scale; let n=a/scale;
  return vec2<f32>(dot(n,d),n.y*d.x-n.x*d.y)/dot(d,d);
}
// Horner value / derivative, with a relative backward error denominator.
fn evaluate(c:array<f32,11>,degree:u32,z:vec2<f32>)->array<vec2<f32>,3> {
  var p=vec2<f32>(c[0],0.0); var dp=vec2<f32>(0.0); var bound=abs(c[0]);
  for(var i=1u;i<=degree;i++) { dp=cmul(dp,z)+p; p=cmul(p,z)+vec2<f32>(c[i],0.0); bound=bound*length(z)+abs(c[i]); }
  return array<vec2<f32>,3>(p,dp,vec2<f32>(bound,0.0));
}

@compute @workgroup_size(32)
fn roots(@builtin(global_invocation_id) id:vec3<u32>) {
  // Exact per-trial bindings; guard the final partial workgroup before any access.
  if(id.x>=params.count) { return; }
  let ob=id.x*176u; let ab=id.x*352u;
  results[ob+13u]=algebra_full[ab+350u];
  if(algebra_full[ab+350u]!=0.0) { results[ob]=algebra_full[ab+350u]; return; }
  var maximum=0.0;
  for(var i=0u;i<11u;i++) { let x=algebra_full[ab+339u+i]; if(!finite_r(x)) { results[ob+15u]=1.0; return; } maximum=max(maximum,abs(x)); }
  if(maximum==0.0) { results[ob+15u]=1.0; return; }
  // Degree reduction is explicit and reported, not silent success at degree ten.
  var first=0u;
  for(var i=0u;i<10u;i++) { if(abs(algebra_full[ab+339u+i])>1e-12*maximum) { break; } first++; }
  let degree=10u-first; results[ob+2u]=f32(degree); results[ob+9u]=f32(first);
  if(degree==0u) { results[ob+15u]=1.0; return; }
  var original:array<f32,11>;
  for(var i=0u;i<=degree;i++) { original[i]=algebra_full[ab+339u+first+i]/maximum; }
  // Fujiwara radius; z=radius*w. Division in steps avoids radius^10 overflow.
  var radius=0.0;
  for(var i=1u;i<=degree;i++) { let ratio=abs(original[i]/original[0]); if(ratio>0.0) { radius=max(radius,2.0*pow(ratio,1.0/f32(i))); } }
  radius=max(radius,1e-6);
  var c:array<f32,11>; c[0]=1.0;
  for(var i=1u;i<=degree;i++) { var x=original[i]/original[0]; for(var j=0u;j<i;j++) { x/=radius; } c[i]=x; }
  var z:array<vec2<f32>,10>; var next:array<vec2<f32>,10>; var done:array<bool,10>;
  for(var i=0u;i<degree;i++) {
    let angle=6.28318530718*(f32(i)+0.37)/f32(degree);
    z[i]=vec2<f32>(cos(angle),sin(angle))*0.75;
  }
  // Jacobi-style simultaneous Aberth updates; never an unbounded root loop.
  for(var iteration=0u;iteration<384u;iteration++) {
    var all_done=true;
    for(var i=0u;i<degree;i++) {
      let ev=evaluate(c,degree,z[i]);
      if(!finite_v(ev[0]) || !finite_v(ev[1]) || !finite_r(ev[2].x) || !(ev[2].x>=0.0)) { results[ob+15u]=1.0; return; }
      let newton=cdiv(ev[0],ev[1]);
      var repulsion=vec2<f32>(0.0);
      for(var j=0u;j<degree;j++) { if(j!=i) { repulsion+=cdiv(vec2<f32>(1.0,0.0),z[i]-z[j]); } }
      var step=cdiv(newton,vec2<f32>(1.0,0.0)-cmul(newton,repulsion));
      if(!finite_v(step)) { results[ob+15u]=1.0; return; }
      let size=length(step); if(size>0.5*(1.0+length(z[i]))) { step*=0.5*(1.0+length(z[i]))/size; }
      next[i]=z[i]-step;
      let updated=evaluate(c,degree,next[i]);
      done[i]=length(step)*radius<=2e-6*(1.0+length(next[i])*radius)
          && length(updated[0])<=2e-6*max(updated[2].x,1e-30)
          && finite_r(next[i].x) && finite_r(next[i].y);
      all_done=all_done && done[i];
    }
    for(var i=0u;i<degree;i++) { z[i]=next[i]; }
    results[ob+3u]=f32(iteration+1u);
    if(all_done) { break; }
  }
  for(var i=0u;i<degree;i++) {
    let sb=ob+16u+i*16u; var root=z[i]*radius;
    let ev=evaluate(c,degree,z[i]); let backward=length(ev[0])/max(ev[2].x,1e-30);
    let final_eval= evaluate(c,degree,vec2<f32>(root.x/radius,0.0));
    if(!finite_v(final_eval[0]) || !finite_r(final_eval[2].x) || length(final_eval[0])>2e-5*max(final_eval[2].x,1e-30)) { results[sb]=3.0; results[ob+4u]+=1.0; continue; }
    results[sb+1u]=root.x; results[sb+2u]=root.y; results[sb+3u]=backward;
    results[ob+10u]=max(results[ob+10u],backward);
    // Distinct from the real-axis residual gate above and the polish gate below.
    if(!done[i]) { results[sb]=7.0; results[ob+4u]+=1.0; continue; }
    // f32 relative realness, deliberately not the CPU's absolute 1e-10 test.
    if(abs(root.y)>2e-4*(1.0+abs(root.x))) { results[sb]=2.0; results[ob+5u]+=1.0; continue; }
    // Real Newton polish, still in scaled coordinates; bounded and guarded.
    var real=z[i].x;
    for(var j=0u;j<8u;j++) {
      let value=evaluate(c,degree,vec2<f32>(real,0.0));
      if(abs(value[1].x)<1e-25) { break; }
      let step=value[0].x/value[1].x;
      if(!finite_r(step) || abs(step)>0.25*(1.0+abs(real))) { break; }
      let candidate=real-step;
      let improved=evaluate(c,degree,vec2<f32>(candidate,0.0));
      // Do not let f32 cancellation turn polishing into a worse root.
      if(abs(improved[0].x)/max(improved[2].x,1e-30)>abs(value[0].x)/max(value[2].x,1e-30)) { break; }
      real=candidate;
    }
    let polished=evaluate(c,degree,vec2<f32>(real,0.0));
    if(abs(polished[0].x)>2e-5*max(polished[2].x,1e-30)) { results[sb]=8.0; results[ob+4u]+=1.0; continue; }
    root=vec2<f32>(real*radius,0.0); results[sb+1u]=root.x; results[sb+2u]=0.0;
    // Close-root clusters are unresolved in f32, not certified multiplicities.
    var duplicate=false;
    for(var j=0u;j<i;j++) {
      let previous=ob+16u+j*16u;
      if(results[previous]==10.0 && abs(results[previous+1u]-root.x)<=2e-3*(1.0+abs(root.x))) { duplicate=true; }
    }
    if(duplicate) { results[sb]=4.0; results[ob+6u]+=1.0; } else { results[sb]=10.0; }
  }
}

// Small symmetric Jacobi SVD via B^T B, scaled first. Returns vector and residual.
fn null3(b:mat3x3<f32>)->vec4<f32> {
  var s=transpose(b)*b; var v=mat3x3<f32>(vec3<f32>(1,0,0),vec3<f32>(0,1,0),vec3<f32>(0,0,1));
  var converged=false;
  for(var sweep=0u;sweep<24u;sweep++) {
    for(var p=0u;p<2u;p++) { for(var q=p+1u;q<3u;q++) {
      let apq=s[q][p]; if(abs(apq)<1e-8) { continue; }
      let tau=(s[q][q]-s[p][p])/(2.0*apq);
      let t=select(-1.0,1.0,tau>=0.0)/(abs(tau)+sqrt(1.0+tau*tau)); let c=inverseSqrt(1.0+t*t); let z=t*c;
      let app=s[p][p]; let aqq=s[q][q];
      for(var k=0u;k<3u;k++) {
        if(k!=p && k!=q) { let x=s[p][k]; let y=s[q][k]; s[p][k]=c*x-z*y; s[k][p]=s[p][k]; s[q][k]=z*x+c*y; s[k][q]=s[q][k]; }
        let x=v[p][k]; let y=v[q][k]; v[p][k]=c*x-z*y; v[q][k]=z*x+c*y;
      }
      s[p][p]=app-t*apq; s[q][q]=aqq+t*apq; s[p][q]=0.0; s[q][p]=0.0;
    }}
    if(max(abs(s[1][0]),max(abs(s[2][0]),abs(s[2][1])))<2e-7) { converged=true; break; }
  }
  if(!converged) { return vec4<f32>(0,0,0,-1); }
  var smallest=0u; for(var i=1u;i<3u;i++) { if(s[i][i]<s[smallest][smallest]) { smallest=i; } }
  let x=normalize(v[smallest]);
  let norm=sqrt(dot(b[0],b[0])+dot(b[1],b[1])+dot(b[2],b[2]));
  if(!(norm>1e-12)) { return vec4<f32>(0,0,0,-1); }
  return vec4<f32>(x,length(b*x)/norm);
}

@compute @workgroup_size(1)
fn recover(@builtin(workgroup_id) id:vec3<u32>) {
  let ob=id.x*176u; let ab=id.x*352u; let db=id.x*40u;
  results[ob+11u]=diagnostics[db+39u]; results[ob+12u]=diagnostics[db+37u];
  if(diagnostics[db+36u]!=0.0) { results[ob]=diagnostics[db+36u]; return; }
  if(results[ob]!=0.0 || results[ob+15u]!=0.0) { return; }
  for(var slot=0u;slot<10u;slot++) {
    let sb=ob+16u+slot*16u;
    if(results[sb]!=10.0) { continue; }
    let z=results[sb+1u]; var bz:mat3x3<f32>;
    for(var r=0u;r<3u;r++) {
      for(var col=0u;col<3u;col++) {
        let start=select(col*4u,8u,col==2u); let count=select(4u,5u,col==2u);
        var x=0.0; for(var k=0u;k<count;k++) { x=x*z+algebra_full[ab+300u+r*13u+start+k]; } bz[col][r]=x;
      }
    }
    var scale=0.0; for(var col=0u;col<3u;col++) { scale=max(scale,max(abs(bz[col].x),max(abs(bz[col].y),abs(bz[col].z)))); }
    if(!(scale>0.0) || !finite_r(scale)) { results[sb]=5.0; results[ob+7u]+=1.0; continue; }
    if(!finite_r(z) || !finite_r(scale)) { results[sb]=6.0; results[ob+7u]+=1.0; continue; }
    let nv=null3(bz*(1.0/scale)); results[sb+13u]=nv.w;
    if(!finite_r(nv.x) || !finite_r(nv.y) || !finite_r(nv.z) || !finite_r(nv.w) || !(nv.w>=0.0 && nv.w<=2e-3) || abs(nv.z)<1e-6) { results[sb]=5.0; results[ob+7u]+=1.0; continue; }
    var e:array<f32,9>; var norm=0.0;
    for(var i=0u;i<9u;i++) {
      e[i]=basis_full[id.x*36u+i]*(nv.x/nv.z)+basis_full[id.x*36u+9u+i]*(nv.y/nv.z)+basis_full[id.x*36u+18u+i]*z+basis_full[id.x*36u+27u+i]; norm+=e[i]*e[i];
    }
    if(!(norm>1e-24) || !finite_r(norm)) { results[sb]=6.0; results[ob+7u]+=1.0; continue; }
    for(var i=0u;i<9u;i++) { e[i]*=inverseSqrt(norm); }
    let matrix=mat3x3<f32>(vec3<f32>(e[0],e[3],e[6]),vec3<f32>(e[1],e[4],e[7]),vec3<f32>(e[2],e[5],e[8]));
    let cubic=2.0*matrix*transpose(matrix)*matrix-matrix;
    let essential_residual=sqrt(dot(cubic[0],cubic[0])+dot(cubic[1],cubic[1])+dot(cubic[2],cubic[2]));
    var residual=0.0; var anorm=0.0;
    for(var r=0u;r<5u;r++) { var x=0.0; for(var j=0u;j<9u;j++) { let a=constraints_full[id.x*45u+r*9u+j]; x+=a*e[j]; anorm+=a*a; } residual+=x*x; }
    residual=sqrt(residual/max(anorm,1e-30));
    results[sb+14u]=residual; results[sb+15u]=essential_residual;
    if(!finite_r(residual) || !finite_r(essential_residual) || !(residual<=5e-4 && essential_residual<=2e-2)) { results[sb]=6.0; results[ob+7u]+=1.0; continue; }
    var duplicate=false;
    for(var j=0u;j<slot;j++) {
      let previous=ob+16u+j*16u;
      if(results[previous]!=1.0) { continue; }
      var minus=0.0; var plus=0.0;
      for(var k=0u;k<9u;k++) { let a=results[previous+4u+k]; minus+=(a-e[k])*(a-e[k]); plus+=(a+e[k])*(a+e[k]); }
      if(min(minus,plus)<4e-8) { duplicate=true; }
    }
    if(duplicate) { results[sb]=9.0; results[ob+8u]+=1.0; continue; }
    results[sb]=1.0; results[ob+1u]+=1.0; results[ob+14u]=max(results[ob+14u],residual);
    for(var k=0u;k<9u;k++) { results[sb+4u+k]=e[k]; }
  }
}
