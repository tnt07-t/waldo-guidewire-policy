import AppKit
let W=1600, H=1850
let rep=NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:W,pixelsHigh:H,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:0,bitsPerPixel:0)!
NSGraphicsContext.saveGraphicsState();NSGraphicsContext.current=NSGraphicsContext(bitmapImageRep:rep)
NSColor.white.setFill();NSRect(x:0,y:0,width:W,height:H).fill()
func text(_ s:String,_ x:CGFloat,_ y:CGFloat,_ w:CGFloat,_ size:CGFloat=28,_ bold:Bool=false){ let p=NSMutableParagraphStyle();p.lineSpacing=5; (s as NSString).draw(in:NSRect(x:x,y:CGFloat(H)-y-120,width:w,height:120),withAttributes:[.font:bold ? NSFont.boldSystemFont(ofSize:size):NSFont.systemFont(ofSize:size),.foregroundColor:NSColor(calibratedWhite:0.13,alpha:1),.paragraphStyle:p]) }
func box(_ s:String,_ x:CGFloat,_ y:CGFloat,_ w:CGFloat,_ h:CGFloat,_ train:Bool=false){let r=NSRect(x:x,y:CGFloat(H)-y-h,width:w,height:h);(train ? NSColor(calibratedRed:0.87,green:0.94,blue:0.9,alpha:1):NSColor(calibratedRed:0.9,green:0.94,blue:0.98,alpha:1)).setFill();NSBezierPath(roundedRect:r,xRadius:13,yRadius:13).fill();text(s,x+15,y+17,w-30,29)}
func arrow(_ x:CGFloat,_ y:CGFloat,_ ex:CGFloat,_ ey:CGFloat){NSColor.darkGray.setStroke();NSColor.darkGray.setFill();let p=NSBezierPath();p.lineWidth=3;p.move(to:NSPoint(x:x,y:CGFloat(H)-y));p.line(to:NSPoint(x:ex,y:CGFloat(H)-ey));p.stroke();let a=NSBezierPath();a.move(to:NSPoint(x:ex,y:CGFloat(H)-ey));a.line(to:NSPoint(x:ex-12,y:CGFloat(H)-ey+8));a.line(to:NSPoint(x:ex-12,y:CGFloat(H)-ey-8));a.close();a.fill()}
text("Five architectural steps • one evolving hypothesis",35,15,1500,40,true)
text("Blue = frozen encoder / fixed inputs     Green = trainable     All outputs: RETRACT or ADVANCE",35,75,1500,28)
let titles=["1–2  Late-goal GRU  |  AdamW changes training, not wiring", "3  Early-goal GRU  |  target now enters every memory update", "4  Latest-image classifier  |  remove visual recurrence", "5  Spatial-grid classifier  |  expose coarse regional detail", "6  Last-block adaptation  |  let action labels change late visual features"]
for k in 0..<5 {let y=CGFloat(145+k*335);text(titles[k],35,y,1530,33,true)
 if k<2 {
 box("6 history images\n+ separate goal",35,y+63,285,115)
 box("Shared frozen ViT\n384 per image",355,y+63,300,115)
 box(k==0 ? "GRU: 386 → 64\nimage + depth/time" : "GRU: 770 → 64\nimage + goal + d/t",690,y+63,400,115,true)
 box("Final state + goal\nMLP: 448 → 64 → 2",1125,y+63,440,115,true)
 arrow(320,y+120,355,y+120);arrow(655,y+120,690,y+120);arrow(1090,y+120,1125,y+120)
 text(k==0 ? "Goal384 bypasses GRU → classifier. Six aligned depth/time pairs enter the GRU steps." : "Goal384 repeats at all six steps AND joins the final classifier; hidden width remains 64.",45,y+196,1500,28)
 text(k==0 ? "Adam: baseline. AdamW: identical 115,650-parameter architecture with weight decay." : "Trainable parameters: 189,378. Earlier goal access also increases capacity.",45,y+244,1500,28)
 } else if k==2 {
 box("Latest image\n+ goal",35,y+63,285,115);box("Shared frozen ViT\n384 + 384",355,y+63,300,115);box("Concatenate + all\n6 depth/time pairs",690,y+63,400,115);box("Trainable classifier\n780 → 64 → 2",1125,y+63,440,115,true)
 arrow(320,y+120,355,y+120);arrow(655,y+120,690,y+120);arrow(1090,y+120,1125,y+120)
 text("No earlier visual frames; numerical movement history remains (12 values).",45,y+196,1500,28);text("50,114 trainable parameters. Each classifier MLP uses ReLU between its two linear layers.",45,y+244,1500,28)
 } else if k==3 {
 box("Latest + goal\nFrozen ViT patches",35,y+63,285,115);box("37×37 → 4×4\n16 regions/image",355,y+63,300,115);box("Shared regional\n384 → 16 + ReLU",690,y+63,400,115,true);box("Flatten + metadata\n524 → 64 → 2",1125,y+63,440,115,true)
 arrow(320,y+120,355,y+120);arrow(655,y+120,690,y+120);arrow(1090,y+120,1125,y+120)
 text("256 latest + 256 goal + 12 depth/time values. Spatial order is retained; no explicit matching.",45,y+196,1500,28);text("39,890 trainable parameters. Frozen patch vectors include context from other patches.",45,y+244,1500,28)
 } else {
 box("Latest + goal\nFrozen ViT prefix",35,y+63,285,115);box("Shared final block\nTRAINABLE",355,y+63,300,115,true);box("Frozen output norm\n384 + 384 summaries",690,y+63,400,115);box("+ metadata12; MLP\n780 → 64 → 2",1125,y+63,440,115,true)
 arrow(320,y+120,355,y+120);arrow(655,y+120,690,y+120);arrow(1090,y+120,1125,y+120)
 text("Adapt attention, internal MLP and norms/scales in the final block; keep the step-4 head.",45,y+196,1500,28);text("Earlier blocks remain frozen. Lower encoder learning rate; no added temporal module.",45,y+244,1500,28)
 }
}
NSGraphicsContext.restoreGraphicsState();try! rep.representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:"tran/figures/architectures.png"))
