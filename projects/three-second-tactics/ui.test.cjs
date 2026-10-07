// Run the actual interface with a small DOM adapter and controllable timers.
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const nodes=new Map();function el(id){if(!nodes.has(id))nodes.set(id,{textContent:'',hidden:true,disabled:false,checked:false,firstChild:{textContent:''},replaceChildren(...v){this.children=v;},append(...v){this.children=[...(this.children||[]),...v];}});return nodes.get(id);}
const ctx=new Proxy({createRadialGradient:()=>({addColorStop(){}})},{get(t,p){return p in t?t[p]:()=>{};},set(t,p,v){t[p]=v;return true;}});el('board').getContext=()=>ctx;el('tutorial-board').getContext=()=>ctx;
const commandNodes=Object.keys(require('./dist/engine.js').actions).map(action=>({dataset:{action}}));let tools={},timers=new Map(),nextId=0;
const doc={getElementById:el,createElement:()=>({append(...v){this.children=[...(this.children||[]),...v];}}),querySelector:el,querySelectorAll:()=>commandNodes,addEventListener(){},modelContext:{registerTool(t){tools[t.name]=t;}}};
const sandbox={document:doc,window:{addEventListener(){}},performance:{now:()=>0},requestAnimationFrame(){},setTimeout(fn,delay){timers.set(++nextId,{fn,delay});return nextId;},clearTimeout(i){timers.delete(i)},AbortController,console};vm.createContext(sandbox);
for(const file of ['engine','renderer','game'])vm.runInContext(fs.readFileSync('dist/'+file+'.js','utf8'),sandbox);
const read=()=>tools.read_match_state.execute();const command=a=>el('commands').onclick({target:{closest:()=>({dataset:{action:a}})}});const select=target=>el('unit-selector').onclick({target:{closest:()=>({dataset:{target:String(target)}})}});
function advance(){const [i,t]=timers.entries().next().value;timers.delete(i);t.fn();}
assert.equal(read().screen,'intro');assert.throws(()=>tools.execute_three_action_plan.execute());tools.enter_match.execute();assert.equal(read().screen,'match');
for(let map=0;map<4;map++){
 el('maps').children[map].onclick();assert.equal(read().state.mapId,sandbox.Tempo.maps[map].id);
 const cmds=[{orders:['right','guard']},{orders:['up','strike']},{orders:['guard','wait']}];tools.set_three_action_plan.execute({actions:cmds});assert.equal(read().plan[1].orders[1],'strike');assert.throws(()=>tools.set_three_action_plan.execute({actions:[{orders:['constructor','up']},...cmds.slice(1)]}));
 assert.equal(el('timeline').children.length,3);assert.equal(el('timeline').children[0].children.length,3);select(1);command('down');assert.equal(read().plan[0].orders[1],'down');assert.equal(read().plan[0].orders[0],'right');assert.equal(read().state.tokens,2);
 el('learning').checked=map%2===1;el('learning').onchange();tools.execute_three_action_plan.execute();assert.equal(timers.values().next().value.delay,map%2===1?1800:1000);
 select(0);command('up');assert.equal(read().state.tokens,1);assert.equal(read().plan[0].orders[1],'down');
 el('show-rules').onclick();assert.equal(timers.size,0);el('show-match').onclick();assert.equal(timers.size,1);
 advance();assert.equal(el('beat-report').children.length,4);advance();advance();advance();assert.equal(read().phase,'plan');assert.equal(read().state.round,2);
 tools.execute_three_action_plan.execute();el('maps').children[(map+1)%4].onclick();assert.equal(timers.size,0);el('maps').children[map].onclick();
 for(let r=0;r<8&&read().phase!=='end';r++){tools.set_three_action_plan.execute({actions:[{orders:['right','right']},{orders:['strike','guard']},{orders:['guard','strike']}]});tools.execute_three_action_plan.execute();while(read().phase==='run'||read().phase==='between')advance();}
 assert.equal(read().phase,'end');el('close-result').onclick();assert.equal(el('result').hidden,true);el('again').onclick();assert.equal(read().phase,'plan');assert.equal(read().state.score[0],0);
}
vm.runInContext(fs.readFileSync('dist/tutorial.js','utf8'),sandbox);assert.equal(el('demo-steps').children.length,3);
sandbox.window.Tutorial.selectStep(0);sandbox.window.Tutorial.tryChoice(1);assert.equal(sandbox.window.Tutorial.getState().state.score[0],0);sandbox.window.Tutorial.tryChoice(0);assert.equal(sandbox.window.Tutorial.getState().state.score[0],1);
sandbox.window.Tutorial.selectStep(1);sandbox.window.Tutorial.tryChoice(0);assert.equal(sandbox.window.Tutorial.getState().state.score[0],0);sandbox.window.Tutorial.tryChoice(1);assert.equal(sandbox.window.Tutorial.getState().state.score[0],1);
sandbox.window.Tutorial.selectStep(2);sandbox.window.Tutorial.tryChoice(1);assert.equal(sandbox.window.Tutorial.getState().state.score[0],0);sandbox.window.Tutorial.tryChoice(0);assert.equal(sandbox.window.Tutorial.getState().state.score[0],2);assert.equal(sandbox.window.Tutorial.getState().completed.length,3);el('demo-toggle').onclick();assert.equal(sandbox.window.Tutorial.getState().resolved,false);
console.log('Independent duo UI passed: 3x2 orders, one-member editing, safe invalid input, interruption retains ally command, four-map matches, pause/replay, and all guided lesson outcomes.');
