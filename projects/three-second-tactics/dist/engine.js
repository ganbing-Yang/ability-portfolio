/* Three-beat simultaneous tactics. This module is shared by the UI and rule checks. */
(function(root){
const actions={up:[0,-1],down:[0,1],left:[-1,0],right:[1,0],strike:null,guard:null,wait:null};
const TARGET=10,ROUNDS=8;
const CENTER=[[5,3],[6,3],[5,4],[6,4]];
const maps=[
 {id:'plaza',name:'风起遗迹',tag:'中央祭坛 · 开阔包夹',tip:'四个金格拼成中央祭坛。两人从相邻位置出发，可以正面占点，也能一起绕到敌人侧面。',width:12,height:8,zones:CENTER,walls:[[4,1],[7,6],[2,5],[9,2]],colors:['#233c48','#79998f','#dfbc72']},
 {id:'bridge',name:'月影门廊',tag:'中央祭坛 · 双门攻防',tip:'两侧石墙把入口收成双门，金格集中在门内。留一人守点，队友从祭坛上、下边缘解围。',width:12,height:8,zones:CENTER,walls:[[4,0],[4,1],[4,2],[4,5],[4,6],[4,7],[7,0],[7,1],[7,2],[7,5],[7,6],[7,7]],colors:['#263449','#7e96b2','#d9c69b']},
 {id:'garden',name:'森灵回廊',tag:'中央祭坛 · 掩体穿插',tip:'金格不再分散到上下两端。石柱位于祭坛外围，可让队友靠近支援，也能借柱子挡住直线冲击。',width:12,height:8,zones:CENTER,walls:[[4,2],[7,5],[4,5],[7,2],[2,2],[9,5]],colors:['#254438','#81a485','#e1c17e']},
 {id:'twins',name:'绯霞环庭',tag:'中央祭坛 · 环绕截击',tip:'两人争夺同一座祭坛。外圈短墙提供两条绕侧路线，尝试一人正面牵制、一人侧面清敌。',width:12,height:8,zones:CENTER,walls:[[5,1],[6,6],[3,2],[8,5],[3,5],[8,2]],colors:['#493647','#ac8d9b','#edc082']}
];
const mapOf=s=>maps.find(m=>m.id===(typeof s==='string'?s:s?.mapId))||maps[0];
const inside=(p,s)=>p.x>=0&&p.x<mapOf(s).width&&p.y>=0&&p.y<mapOf(s).height;
const zone=(p,s)=>mapOf(s).zones.some(([x,y])=>p.x===x&&p.y===y);
const walkable=(p,s)=>inside(p,s)&&!mapOf(s).walls.some(([x,y])=>p.x===x&&p.y===y);
function inRange(p,q,s){const dx=q.x-p.x,dy=q.y-p.y;if(Math.abs(dx)+Math.abs(dy)>2||(dx!==0&&dy!==0))return false;const n=Math.abs(dx)+Math.abs(dy);if(n===0)return false;for(let j=1;j<n;j++)if(!walkable({x:p.x+Math.sign(dx)*j,y:p.y+Math.sign(dy)*j},s))return false;return true;}
function command(c){if(c&&Array.isArray(c.orders)){if(c.orders.length!==2||c.orders.some(a=>!Object.hasOwn(actions,a)))throw Error('每拍需要两名角色各一个有效动作');return {orders:[...c.orders]};}const v=typeof c==='string'?{target:0,action:c}:c;if(!v||!Object.hasOwn(actions,v.action)||![0,1,'all'].includes(v.target))throw Error('无效指令');return {orders:[0,1].map(i=>v.target==='all'||v.target===i?v.action:'wait')};}
function expand(c){return command(c).orders;}
const pressured=(p,s)=>s.units.some(q=>q.team!==p.team&&Math.abs(q.x-p.x)+Math.abs(q.y-p.y)===1);
const canScore=(p,s)=>zone(p,s)&&!pressured(p,s);
function fresh(mapId='plaza'){if(!maps.some(m=>m.id===mapId))throw Error('Unknown map');const s={mapId,units:[{x:3,y:3,team:0,name:'青岚'},{x:3,y:4,team:0,name:'月汐'},{x:8,y:4,team:1,name:'绯烬'},{x:8,y:3,team:1,name:'星焰'}],score:[0,0],tokens:2,round:1,beat:0,interrupts:0};s.units.forEach(p=>{p.shield=2;p.steady=0;});return s;}
const same=(p,q)=>p.x===q.x&&p.y===q.y;
function settle(old,desired,events,label){let p=desired.map(v=>({...v}));for(let pass=0;pass<old.length+1;pass++){const blocked=new Set();for(let i=0;i<p.length;i++)for(let j=i+1;j<p.length;j++){if(same(p[i],p[j])||(same(p[i],old[j])&&same(p[j],old[i]))){blocked.add(i);blocked.add(j);}}let changed=false;for(const i of blocked)if(!same(p[i],old[i])){p[i]={...old[i]};events?.[i].push(label);changed=true;}if(!changed)break;}return p;}
function resolve(s,pc,ec){
 const a=[...expand(pc),...expand(ec)],old=s.units.map(p=>({...p})),events=old.map(()=>[]);
 const desired=old.map((p,i)=>{const d=actions[a[i]];if(!d)return {...p};const q={...p,x:p.x+d[0],y:p.y+d[1]};events[i].push(walkable(q,s)?'移动 1 格':'移动被边界或石墙挡住');return walkable(q,s)?q:{...p};});
 const moved=settle(old,desired,events,'道路被角色阻挡，停在原位'),pushes=old.map(()=>[]),hits=[],guardActive=old.map((p,i)=>a[i]==='guard'&&p.shield>0),protectedNow=old.map(p=>p.steady>0);
 moved.forEach((p,i)=>{p.shield=a[i]==='guard'?Math.max(0,p.shield-1):Math.min(2,p.shield+1);p.steady=Math.max(0,p.steady-1);if(a[i]==='guard')events[i].push(guardActive[i]?'防守消耗 1 点护盾':'护盾耗尽，本拍无法格挡');});
 for(let i=0;i<4;i++){
 if(a[i]==='strike'){
 const candidates=moved.map((p,j)=>({p,j,d:Math.abs(p.x-moved[i].x)+Math.abs(p.y-moved[i].y)})).filter(t=>t.p.team!==moved[i].team&&inRange(moved[i],t.p,s)).sort((x,y)=>x.d-y.d||x.j-y.j);
 if(!candidates.length){events[i].push('冲击落空：敌人不在无遮挡的直线 2 格内');continue;}
 const j=candidates[0].j;if(guardActive[j]){events[i].push('冲击被防守挡住');events[j].push('成功格挡');hits.push({from:i,to:j,blocked:true});continue;}
 if(protectedNow[j]){events[i].push('命中，但敌人处于稳身保护，不能击退');events[j].push('稳身保护生效');hits.push({from:i,to:j,blocked:false,protected:true});continue;}
 pushes[j].push([Math.sign(moved[j].x-moved[i].x),Math.sign(moved[j].y-moved[i].y)]);hits.push({from:i,to:j,blocked:false});
 }else events[i].push(a[i]==='guard'?'原地防守':a[i]==='wait'?'原地等待':'完成移动');
 }
 const pushed=moved.map((p,i)=>{if(!pushes[i].length)return {...p};const dx=Math.sign(pushes[i].reduce((v,d)=>v+d[0],0)),dy=Math.sign(pushes[i].reduce((v,d)=>v+d[1],0));if(dx&&dy){events[i].push('多个方向的冲击抵消');return {...p};}const q={...p,x:p.x+dx,y:p.y+dy};return walkable(q,s)?q:{...p};});
 const pos=settle(moved,pushed,events,'击退路线被角色挡住');for(const h of hits)if(!h.blocked){h.pushed=!same(pos[h.to],moved[h.to]);if(h.pushed)pos[h.to].steady=1;if(!h.protected)events[h.from].push(h.pushed?'命中，击退敌人 1 格；对方下一拍稳身':'命中，但障碍或其他冲击阻止了击退');}
 s.units=pos;const gain=pos.map(p=>canScore(p,s)?1:0),sideGain=[gain[0]+gain[1],gain[2]+gain[3]];gain.forEach((v,i)=>events[i].push(v?'据点无人贴身压制：+1 分':zone(pos[i],s)?'敌人贴身压制：据点 +0 分':'不在据点：+0 分'));s.score=s.score.map((v,i)=>v+sideGain[i]);s.beat++;return {old,pos,hits,gain,sideGain,events,actions:a,guardActive};
}
function interrupt(s,plan,index,c){if(!Number.isInteger(index)||index<s.beat||index>2||s.tokens<=0)return false;const next=command(c),prev=command(plan[index]);if(next.orders.every((a,i)=>a===prev.orders[i]))return false;plan[index]=next;s.tokens--;s.interrupts++;return true;}
function finishRound(s){if(s.interrupts===0)s.tokens=Math.min(3,s.tokens+1);s.round++;s.beat=0;s.interrupts=0;}
function winner(s){if(Math.max(...s.score)>=TARGET||s.round>ROUNDS)return s.score[0]===s.score[1]?'draw':s.score[0]>s.score[1]?'player':'enemy';return null;}
function route(p,s){const queue=[{p,path:[]}],seen=new Set([p.x+','+p.y]);while(queue.length){const curr=queue.shift();if(zone(curr.p,s))return curr.path;for(const a of ['up','down','left','right']){const d=actions[a],q={x:curr.p.x+d[0],y:curr.p.y+d[1]},key=q.x+','+q.y;if(walkable(q,s)&&!seen.has(key)){seen.add(key);queue.push({p:q,path:[...curr.path,a]});}}}return [];}
function enemyPlan(s,random=Math.random){const result=[],future=JSON.parse(JSON.stringify(s));for(let beat=0;beat<3;beat++){
 let best={target:'all',action:'wait'},bestValue=-Infinity;
 for(const first of Object.keys(actions))for(const second of Object.keys(actions)){const c={orders:[first,second]},test=JSON.parse(JSON.stringify(future)),before=future.units.slice(2).map(p=>route(p,future).length),effect=resolve(test,{target:'all',action:'wait'},c);let value=effect.sideGain[1]*5-effect.sideGain[0]*3+effect.hits.filter(h=>h.from>=2&&h.pushed).length*2;
 for(let j=2;j<4;j++)value+=(before[j-2]-route(test.units[j],test).length)*.65;
 value+=future.units.slice(2).filter((p,i)=>effect.guardActive[i+2]&&future.units.slice(0,2).some(q=>inRange(p,q,future))).length*.7;value+=test.units.slice(2).reduce((v,p)=>v+p.shield,0)*.06;
 value+=random()*.12;if(value>bestValue){bestValue=value;best=c;}}
 result.push(best);resolve(future,{target:'all',action:'wait'},best);
 }return result;}
function defaultPlan(s){const steps=route(s.units[0],s).slice(0,3);while(steps.length<3)steps.push('guard');const other=route(s.units[1],s).slice(0,3);while(other.length<3)other.push('guard');return steps.map((action,i)=>({orders:[action,other[i]]}));}
const api={TARGET,ROUNDS,CENTER,actions,maps,mapOf,inside,zone,pressured,canScore,walkable,inRange,command,expand,settle,fresh,resolve,interrupt,finishRound,winner,route,enemyPlan,defaultPlan};if(typeof module!=='undefined')module.exports=api;else root.Tempo=api;
})(globalThis);
