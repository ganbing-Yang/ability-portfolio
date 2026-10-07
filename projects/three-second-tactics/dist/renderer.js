/* Orthographic 3D meshes, depth ordering and face lighting, rendered without remote assets. */
(function(root){
const SX=42,SY=20,U=73,OX=406,OY=103;
const project=([x,z,h=0])=>[OX+(x-z)*SX,OY+(x+z)*SY-h*U];
function color(hex,n){const c=hex.replace('#','');return '#'+[0,2,4].map(i=>Math.min(255,Math.max(0,Math.round(parseInt(c.slice(i,i+2),16)*n))).toString(16).padStart(2,'0')).join('');}
function face(list,points,col){const depth=points.reduce((n,v)=>n+v[0]+v[1]+v[2]*.55,0)/points.length;list.push({points,col,depth});}
function box(list,x,z,h,w,d,height,col){const a=[x-w/2,z-d/2,h-height/2],b=[x+w/2,z+d/2,h+height/2];const v=[[a[0],a[1],a[2]],[b[0],a[1],a[2]],[b[0],b[1],a[2]],[a[0],b[1],a[2]],[a[0],a[1],b[2]],[b[0],a[1],b[2]],[b[0],b[1],b[2]],[a[0],b[1],b[2]]];[[[0,1,5,4],.72],[[1,2,6,5],.86],[[2,3,7,6],.95],[[3,0,4,7],.74],[[4,5,6,7],1.2]].forEach(([ids,n])=>face(list,ids.map(i=>v[i]),color(col,n)));}
function ellipsoid(list,x,z,h,rx,rz,rh,col,limit=Math.PI){const rings=7,segs=12;for(let j=0;j<rings;j++)for(let i=0;i<segs;i++){const vertex=(k,l)=>{const p=k/rings*limit,t=l/segs*Math.PI*2;return [x+rx*Math.sin(p)*Math.cos(t),z+rz*Math.sin(p)*Math.sin(t),h+rh*Math.cos(p)];};const mid=(j+.5)/rings*limit,t=(i+.5)/segs*Math.PI*2;const shade=.77+.18*Math.cos(mid)-.12*Math.sin(mid)*Math.cos(t)-.08*Math.sin(mid)*Math.sin(t);face(list,[vertex(j,i),vertex(j,i+1),vertex(j+1,i+1),vertex(j+1,i)],color(col,shade));}}
function paint(ctx,list){list.sort((a,b)=>a.depth-b.depth);for(const f of list){ctx.beginPath();f.points.forEach((p,i)=>{const [x,y]=project(p);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.closePath();ctx.fillStyle=f.col;ctx.fill();ctx.strokeStyle=f.col;ctx.lineWidth=.45;ctx.stroke();}}
function diamond(ctx,x,z,inset,fill,stroke){const points=[[x+inset,z+inset,.09],[x+1-inset,z+inset,.09],[x+1-inset,z+1-inset,.09],[x+inset,z+1-inset,.09]].map(project);ctx.beginPath();points.forEach(([a,b],i)=>i?ctx.lineTo(a,b):ctx.moveTo(a,b));ctx.closePath();if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke();}}
function knight(list,x,z,i,pose,time,variant=0){
 const accent=i?'#ff886c':'#77f1dc',cloth=i?(variant?'#683953':'#503756'):(variant?'#405078':'#244963'),hair=i?(variant?'#b7754b':'#a54348'):(variant?'#95afc7':'#355b72'),metal=i?'#ffd2a1':'#dfedf4';
 const moving=pose==='move',attacking=pose==='strike',bob=moving?Math.sin(time/70)*.035:Math.sin(time/420)*.008,lean=attacking?Math.sin(time/100)*.09:0,base=.11+bob;x+=lean;
 // Long split coat, high collar, scarf and asymmetrical sleeves give each hero a distinct silhouette.
 box(list,x-.12,z,base+.12,.15,.23,.23,'#273040');box(list,x+.12,z,base+.12,.15,.23,.23,'#273040');
 box(list,x,z,base+.42,.33,.26,.44,cloth);box(list,x-.105,z+.025,base+.255,.16,.3,.24,cloth);box(list,x+.105,z+.025,base+.255,.16,.3,.24,cloth);
 box(list,x,z+.145,base+.5,.21,.025,.26,i?'#aa6371':'#eef2e6');box(list,x,z+.169,base+.38,.35,.03,.055,'#bd9d70');box(list,x+.13,z+.19,base+.39,.06,.03,.08,accent);
 box(list,x,z,base+.7,.4,.3,.09,cloth);box(list,x,z+.17,base+.685,.31,.04,.065,accent);
 face(list,[[x-.17,z-.14,base+.73],[x+.15,z-.14,base+.73],[x+.34,z-.45,base+.42],[x+.24,z-.43,base+.18],[x-.16,z-.29,base+.23]],i?'#bf5c63':'#54bfae');
 box(list,x-.24,z,base+.5,.13,.17,.31,cloth);box(list,x+.24,z,base+.5,.13,.17,.31,cloth);ellipsoid(list,x-.24,z+.03,base+.345,.067,.07,.075,'#f0ccb9');ellipsoid(list,x+.24,z+.03,base+.345,.067,.07,.075,'#f0ccb9');
 // Anime proportions: large expressive eyes, exposed face and angular sculpted bangs.
 ellipsoid(list,x,z,base+1.035,.26,.235,.29,'#f0d5c1');ellipsoid(list,x,z-.035,base+1.12,.29,.255,.28,hair,1.85);
 box(list,x,z+.22,base+1.035,.32,.035,.21,'#f3dbc9');
 for(const side of [-1,1]){const ex=x+side*.081;box(list,ex,z+.248,base+1.055,.099,.012,.081,'#fff4e8');box(list,ex+side*.007,z+.26,base+1.055,.05,.014,.073,accent);box(list,ex+side*.01,z+.27,base+1.055,.023,.008,.058,'#263447');box(list,ex-.013,z+.279,base+1.08,.018,.008,.018,'#ffffff');box(list,ex,z+.252,base+1.108,.112,.015,.014,hair);box(list,ex+side*.02,z+.252,base+.986,.054,.009,.017,'#e9b5ab');}
 box(list,x,z+.253,base+.947,.065,.013,.014,'#a57075');
 for(let n=0;n<5;n++){const hx=x-.25+n*.12;face(list,[[hx,z+.18,base+1.23],[hx+.13,z+.19,base+1.21],[hx+.08,z+.278,base+1.065+(n%2)*.06]],n%2?color(hair,1.15):hair);}
 for(let n=0;n<4;n++){const hx=x-.22+n*.14;face(list,[[hx,z-.035,base+1.32],[hx+.15,z-.08,base+1.27],[hx+.055,z-.05,base+1.49-(n%2)*.07]],color(hair,1.12));}
 // A wind sword and a flame spear share the same push mechanic, with different visual identities.
 if(!i){const wx=x+.37,zz=z+.06;box(list,wx,zz,base+.42,.067,.08,.21,'#43596b');box(list,wx,zz,base+.54,.26,.065,.045,accent);face(list,[[wx-.065,zz,base+.56],[wx+.065,zz,base+.56],[wx+.055,zz,base+1.15],[wx,zz,base+1.38],[wx-.055,zz,base+1.15]],metal);box(list,wx,zz+.012,base+.91,.02,.016,.62,accent);}
 else{const wx=x+.38,zz=z+.06;box(list,wx,zz,base+.71,.045,.045,1.35,'#765b66');face(list,[[wx-.09,zz,base+1.33],[wx+.09,zz,base+1.33],[wx,zz,base+1.62]],metal);face(list,[[wx-.045,zz+.008,base+1.36],[wx+.045,zz+.008,base+1.36],[wx,zz+.008,base+1.56]],accent);face(list,[[wx-.04,zz,base+1.12],[wx+.23,zz,base+1.02],[wx+.18,zz,base+.77],[wx+.02,zz,base+.88]],'#d66565');}
}

function scenery(list,map,time){
 const points=[[-.55,1],[-.55,map.height-2],[2,-.55],[map.width-3,-.55],[map.width+.55,2],[map.width+.55,map.height-2]];
 for(let i=0;i<points.length;i++){const [x,z]=points[i];
 if(map.id==='garden'){box(list,x,z,.27,.12,.12,.54,'#6c6057');ellipsoid(list,x,z,.86,.36,.35,.54,i%2?'#487d68':'#608f6d');ellipsoid(list,x+.15,z+.07,.55,.26,.25,.3,'#71a37c');}
 else if(map.id==='bridge'){box(list,x,z,.27,.12,.12,.54,'#586880');box(list,x,z,.65,.29,.29,.25,'#a4c4db');box(list,x,z,.84,.37,.37,.09,'#536f91');}
 else if(map.id==='twins'){box(list,x,z,.64,.07,.07,1.28,'#6f586b');face(list,[[x,z,.99],[x+.36,z,1.06],[x+.34,z,.54],[x,z,.66]],i%2?'#d79081':'#987695');}
 else{box(list,x,z,.22,.44,.44,.44,'#81928e');box(list,x,z,.63,.28,.28,.38,'#a2b1a5');box(list,x,z,.84,.39,.39,.07,'#bcc8b4');}
 }
}

function render(ctx,{state,plan,phase,selected,animation,effect,poses,time,selectedUnit=0}){const T=root.Tempo,map=T.mapOf(state);ctx.clearRect(0,0,980,620);const bg=ctx.createRadialGradient(490,310,30,490,310,620);bg.addColorStop(0,map.colors[0]);bg.addColorStop(1,'#0c1522');ctx.fillStyle=bg;ctx.fillRect(0,0,980,620);
 // The floating arena has thickness and bevel highlights; the layout remains a 7x5 grid.
 const island=[];box(island,map.width/2,map.height/2,-.18,map.width+.25,map.height+.25,.4,'#263442');paint(ctx,island);const ground=[];for(let z=0;z<map.height;z++)for(let x=0;x<map.width;x++){const scoring=T.zone({x,y:z},state);box(ground,x+.5,z+.5,0,.94,.94,.16,scoring?map.colors[2]:((x+z)%2?color(map.colors[1],.95):map.colors[1]));}paint(ctx,ground);
 const outline=[[5,3,.14],[7,3,.14],[7,5,.14],[5,5,.14]].map(project);ctx.beginPath();outline.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.strokeStyle='#ffe7a3';ctx.lineWidth=3;ctx.shadowColor='#f8d187';ctx.shadowBlur=14;ctx.stroke();ctx.shadowBlur=0;
 for(const [x,z]of map.zones){diamond(ctx,x,z,.14,'#fff4bb22','#fff2c1');const [px,py]=project([x+.5,z+.5,.13]);ctx.fillStyle='#fff3cd';ctx.font='bold 16px system-ui';ctx.textAlign='center';const contested=state.units.some(p=>p.x===x&&p.y===z&&T.pressured(p,state));if(contested)diamond(ctx,x,z,.14,'#da715355','#f7a087');ctx.fillStyle=contested?'#ffddd0':'#fff3cd';ctx.fillText(contested?'暂停':'+1',px,py+5);}
 if(phase==='plan'){const preview=state.units.slice(0,2).map(p=>({...p}));const marks=[];for(let beat=0;beat<3;beat++){const c=T.command(plan[beat]),a=T.expand(c),before=preview.map(p=>({...p})),desired=preview.map((p,j)=>{const d=T.actions[a[j]],q=d?{...p,x:p.x+d[0],y:p.y+d[1]}:{...p};return T.walkable(q,state)?q:{...p};}),next=T.settle(before,desired);for(let j=0;j<2;j++){const active=true;if(active){const blocked=Boolean(T.actions[a[j]])&&next[j].x===before[j].x&&next[j].y===before[j].y;marks.push({p:next[j],beat,j,blocked});if(beat===selected&&a[j]==='strike')for(let z=0;z<map.height;z++)for(let x=0;x<map.width;x++)if(T.inRange(next[j],{x,y:z},state))diamond(ctx,x,z,.2,'#ffb68b44','#ffc8a0');}preview[j]=next[j];}}for(const step of marks){const [px,py]=project([step.p.x+.5,step.p.y+.5,.17]);ctx.beginPath();ctx.ellipse(px,py,18,10,0,0,Math.PI*2);ctx.fillStyle=step.blocked?'#d47864':step.beat===selected?'#a1f9de':'#274b49';ctx.fill();ctx.fillStyle=step.beat===selected?'#163d36':'#e0fff6';ctx.font='bold 11px system-ui';ctx.fillText(step.blocked?'×':`${step.j+1}·${step.beat+1}`,px,py+4);}}

 const scene=[];scenery(scene,map,time);for(const [x,z]of map.walls){box(scene,x+.5,z+.5,.32,.72,.72,.64,'#536776');box(scene,x+.5,z+.5,.67,.79,.79,.09,'#8da3ad');box(scene,x+.5,z+.5,.715,.53,.53,.04,map.colors[1]);}
 const prog=animation?Math.min(1,(time-animation.start)/260):1;const characters=state.units.map((p,i)=>{const from=animation?animation.from[i]:p;return {x:from.x+(p.x-from.x)*prog+.5,z:from.y+(p.y-from.y)*prog+.5};});
 for(let i=0;i<4;i++){const p=characters[i],[px,py]=project([p.x,p.z,.1]);const focus=T.command(plan[Math.min(2,phase==='run'?state.beat:selected)]);if(i<2&&i===(selectedUnit)){ctx.beginPath();ctx.ellipse(px,py+2,27,13,0,0,Math.PI*2);ctx.strokeStyle=i?'#bcd4ff':'#94ffe0';ctx.lineWidth=2;ctx.stroke();}ctx.beginPath();ctx.ellipse(px,py+2,25,11,0,0,Math.PI*2);ctx.fillStyle='#08121e80';ctx.fill();knight(scene,p.x,p.z,i>=2,poses?.[i]||'idle',time,i%2);}paint(ctx,scene);
 for(let i=0;i<4;i++){const p=characters[i],[px,py]=project([p.x,p.z,1.97]);ctx.fillStyle=i>=2?'#ffb998':'#a0ffdf';ctx.font='bold 13px system-ui';ctx.fillText(`${i%2+1} · ${state.units[i].name}${T.zone(state.units[i],state)?T.pressured(state.units[i],state)?' · 被压制':' · +1':''}`,px,py);const [sx,sy]=project([p.x,p.z,1.73]);ctx.font='10px system-ui';ctx.fillStyle=state.units[i].shield?'#b9d8e4':'#ffac8c';ctx.fillText(`盾 ${'◆'.repeat(state.units[i].shield)}${'◇'.repeat(2-state.units[i].shield)}${state.units[i].steady?' · 稳身':''}`,sx,sy);if(state.units[i].steady){const [a,b]=project([p.x,p.z,.62]);ctx.strokeStyle='#f3d9a2aa';ctx.lineWidth=2;ctx.beginPath();ctx.ellipse(a,b,32,44,0,0,Math.PI*2);ctx.stroke();}if(poses?.[i]==='guard'){const [a,b]=project([p.x,p.z,.61]);ctx.strokeStyle=i>=2?'#ffba9990':'#86ffe4aa';ctx.lineWidth=3;ctx.beginPath();ctx.ellipse(a,b,35,49,-.2,0,Math.PI*2);ctx.stroke();}}
 if(effect&&animation&&time-animation.start<750){for(let i=0;i<4;i++)if(effect.gain[i]){const p=characters[i],[px,py]=project([p.x,p.z,2.12+(time-animation.start)/1200]);ctx.fillStyle='#ffedac';ctx.font='bold 21px system-ui';ctx.fillText('+1',px,py);}effect.hits.forEach(h=>{const p=characters[h.to],[px,py]=project([p.x,p.z,1.1]);ctx.fillStyle=h.blocked?'#bcebdc':'#ffd5a8';ctx.font='bold 14px system-ui';ctx.fillText(h.blocked?'格挡！':h.protected?'稳身保护':h.pushed?'击退！':'受阻',px,py-16);});}
 for(let i=0;i<4;i++){const p=characters[i],[px,py]=project([p.x,p.z,.85]);if(poses?.[i]==='strike'){ctx.save();ctx.strokeStyle=i>=2?'#ffb481':'#8effe5';ctx.shadowBlur=18;ctx.shadowColor=ctx.strokeStyle;ctx.lineWidth=5;ctx.beginPath();ctx.ellipse(px,py,43,27,time/170,.1,Math.PI*1.35);ctx.stroke();ctx.restore();}}
 ctx.fillStyle='#a1b3c9';ctx.font='12px system-ui';ctx.fillText(phase==='plan'?'角色号·拍数 = 预计落点   × = 受阻   橙色 = 冲击范围':'移动 → 冲击 / 防守 → 无人贴身才得分',490,572);
 ctx.font='11px system-ui';ctx.fillStyle='#758ba6';ctx.fillText('中央祭坛 · 两人协作    |    W ↗   D ↘   S ↙   A ↖',490,596);
}
root.TempoRenderer={render,project};
})(globalThis);
