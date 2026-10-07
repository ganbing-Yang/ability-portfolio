const assert=require('node:assert/strict'),T=require('./dist/engine.js'),C=(target,action)=>({target,action}),wait={orders:['wait','wait']};
function scenario(p0=[3,3],p1=[1,4],e0=[10,3],e1=[10,4],map='plaza'){const s=T.fresh(map);[p0,p1,e0,e1].forEach(([x,y],i)=>s.units[i]={...s.units[i],x,y});return s;}
let s=T.fresh();assert.equal(s.units.length,4);assert.equal(Math.abs(s.units[0].x-s.units[1].x)+Math.abs(s.units[0].y-s.units[1].y),1);T.resolve(s,C(0,'right'),wait);assert.equal(s.units[0].x,4);assert.equal(s.units[1].x,3);
s=T.fresh();T.resolve(s,C('all','right'),wait);assert.deepEqual(s.units.slice(0,2).map(p=>p.x),[4,4]);
s=scenario([4,3],[4,4]);let r=T.resolve(s,{orders:['right','right']},wait);assert.equal(s.score[0],2);assert.deepEqual(r.sideGain,[2,0]);
s=scenario([4,3],[1,4],[6,3]);T.resolve(s,C(0,'right'),C(0,'left'));assert.deepEqual([s.units[0].x,s.units[2].x],[4,6]);
s=scenario([4,3],[5,3]);T.resolve(s,C(0,'right'),wait);assert.deepEqual(s.units.slice(0,2).map(p=>p.x),[4,5]);
s=scenario([3,3],[4,3],[5,3]);T.resolve(s,C('all','right'),wait);assert.deepEqual(s.units.slice(0,3).map(p=>p.x),[3,4,5]);
s=scenario([4,3],[1,4],[5,3]);T.resolve(s,C(0,'right'),C(0,'left'));assert.deepEqual([s.units[0].x,s.units[2].x],[4,5]);
s=scenario([5,3],[1,4],[6,3]);r=T.resolve(s,C(0,'guard'),C(0,'strike'));assert.equal(s.units[0].x,5);assert.equal(s.score[0],0);assert(r.guardActive[0]);
s=scenario([5,3],[1,4],[6,3]);r=T.resolve(s,C(0,'wait'),C(0,'strike'));assert.equal(s.units[0].x,4);assert.equal(s.score[0],0);assert(r.hits[0].pushed);
s=scenario([7,4],[1,4],[7,5]);r=T.resolve(s,C(0,'strike'),wait);assert.equal(s.units[2].y,5);assert.equal(r.hits[0].pushed,false);
s=scenario([4,3],[1,4],[5,3],[6,3]);r=T.resolve(s,C(0,'strike'),wait);assert.equal(r.hits.length,1);assert.equal(r.hits[0].to,2);assert.equal(s.units[2].x,5);
s=T.fresh();const plan=T.defaultPlan(s);assert(T.interrupt(s,plan,0,{orders:[plan[0].orders[0],'wait']}));assert.equal(s.tokens,1);assert(!T.interrupt(s,plan,0,{orders:[plan[0].orders[0],'wait']}));s.beat=1;assert(!T.interrupt(s,plan,0,C('all','up')));T.finishRound(s);assert.equal(s.tokens,1);
s=T.fresh();T.finishRound(s);assert.equal(s.tokens,3);s.score=[10,10];assert.equal(T.winner(s),'draw');s.score=[10,9];assert.equal(T.winner(s),'player');s.score=[2,3];s.round=9;assert.equal(T.winner(s),'enemy');s.round=8;assert.equal(T.winner(s),null);
s=scenario([5,3],[5,4],[6,3],[6,4]);for(let beat=0;beat<2;beat++){r=T.resolve(s,C('all','guard'),C('all','strike'));assert.equal(r.sideGain[0],0);assert.equal(s.units[0].x,5);assert.equal(s.units[0].shield,1-beat);}r=T.resolve(s,C('all','guard'),C('all','strike'));assert.deepEqual(s.units.slice(0,2).map(p=>p.x),[4,4]);assert.equal(s.units[0].steady,1);
r=T.resolve(s,C('all','right'),C('all','strike'));assert.equal(s.units[0].x,5);assert.equal(s.units[0].steady,0);assert(r.hits.every(h=>h.protected));assert.equal(s.units[0].shield,1);r=T.resolve(s,wait,C('all','strike'));assert.equal(s.units[0].x,4);assert.equal(s.units[0].shield,2);
// A shared altar makes BOTH roles contribute to one same-beat objective.
function cooperation(){return scenario([5,4],[6,3],[6,4],[9,6]);}
s=cooperation();r=T.resolve(s,{orders:['guard','guard']},{orders:['strike','wait']});assert.equal(r.sideGain[0],0);assert(T.pressured(s.units[0],s));
s=cooperation();r=T.resolve(s,{orders:['guard','strike']},{orders:['strike','wait']});assert.equal(r.sideGain[0],2);assert.deepEqual(s.units[2].y,5);assert(!T.pressured(s.units[0],s));assert(!T.pressured(s.units[1],s));
s=cooperation();r=T.resolve(s,{orders:['wait','wait']},{orders:['strike','wait']});assert.equal(r.sideGain[0],0);
assert.throws(()=>T.command({orders:['left']}));assert.throws(()=>T.command({orders:['guard','constructor']}));assert(!('up_right' in T.actions));assert.equal(Object.values(T.actions).filter(Boolean).length,4);
for(const map of T.maps){
 assert.deepEqual(map.zones,[[5,3],[6,3],[5,4],[6,4]]);assert(map.zones.every(([x,y])=>T.walkable({x,y},map.id)));const start=T.fresh(map.id);assert.equal(T.route(start.units[0],start).length,T.route(start.units[2],start).length);assert.equal(T.route(start.units[1],start).length,T.route(start.units[3],start).length);
 for(const p of start.units){const q=[p],seen=new Set([p.x+','+p.y]);while(q.length){const curr=q.shift();for(const d of Object.values(T.actions).filter(Boolean)){const n={x:curr.x+d[0],y:curr.y+d[1]},key=n.x+','+n.y;if(T.walkable(n,map.id)&&!seen.has(key)){seen.add(key);q.push(n);}}}for(const [x,y]of map.zones)assert(seen.has(x+','+y));}
 // Map obstacles are symmetric around the central objective.
 for(const [x,y]of map.walls)assert(map.walls.some(([a,b])=>a===11-x&&b===7-y));
 for(let game=0;game<20;game++){s=T.fresh(map.id);for(let round=0;round<T.ROUNDS;round++){const ep=T.enemyPlan(s);for(let b=0;b<3;b++){const rand=()=>Object.keys(T.actions)[Math.floor(Math.random()*7)];T.resolve(s,{orders:[rand(),rand()]},ep[b]);assert(s.units.every(p=>T.walkable(p,s)&&p.shield>=0&&p.shield<=2&&[0,1].includes(p.steady)));assert.equal(new Set(s.units.map(p=>p.x+','+p.y)).size,4);}T.finishRound(s);}}
}
console.log('Central altar checked: shared 2x2 scoring zone, adjacent teammates, symmetric approach/obstacles, complementary duo scoring, no diagonal movement, shield/pressure/collisions and 80 matches.');
