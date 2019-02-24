<Qucs Schematic 0.0.18>
<Properties>
  <View=146,153,1337,677,1.37154,0,0>
  <Grid=10,10,1>
  <DataSet=S_param_probe.dat>
  <DataDisplay=S_param_probe.dpl>
  <OpenDisplay=1>
  <Script=S_param_probe.m>
  <RunScript=0>
  <showFrame=0>
  <FrameText0=Title>
  <FrameText1=Drawn By:>
  <FrameText2=Date:>
  <FrameText3=Revision:>
</Properties>
<Symbol>
</Symbol>
<Components>
  <Pac P3 1 280 350 18 -26 0 1 "3" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
  <Pac P1 1 200 290 18 -26 0 1 "1" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
  <Pac P4 1 460 350 18 -26 0 1 "4" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
  <GND * 1 200 340 0 0 0 0>
  <GND * 1 280 400 0 0 0 0>
  <GND * 1 460 400 0 0 0 0>
  <GND * 1 560 340 0 0 0 0>
  <Pac P2 1 560 290 18 -26 0 1 "2" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
  <GND * 1 390 350 0 0 0 0>
  <Eqn Eqn1 1 380 480 -25 13 0 0 "Sdd_2_1=0.5*(S[2,1]-S[2,3]+S[4,3]-S[4,1])" 1 "yes" 0>
  <Eqn Eqn2 1 380 540 -25 13 0 0 "impulse=abs(Freq2Time(Sdd_2_1, frequency))" 1 "yes" 0>
  <Eqn Eqn6 1 380 600 -25 13 0 0 "impulse_adj=impulse - 0.525 * impulse[0]" 1 "yes" 0>
  <Eqn Eqn3 1 750 480 -25 13 0 0 "Mag_Sdd21=abs(Sdd_2_1)" 1 "yes" 0>
  <Eqn Eqn4 1 940 480 -25 13 0 0 "Phase_Sdd21=arg(Sdd_2_1)" 1 "yes" 0>
  <Eqn Eqn5 1 750 540 -25 13 0 0 "step=cumsum(impulse)" 1 "yes" 0>
  <Eqn Eqn7 1 750 600 -25 13 0 0 "step_adj=cumsum(impulse_adj)" 1 "yes" 0>
  <SPfile X1 1 390 270 -26 -77 0 0 "/Users/dbanas/Documents/Projects/PyBERT/channels/802.3bj_COM_Cisco/kochuparambil_3bj_02_0913/Beth_shortReflective_THRU.s4p" 1 "rectangular" 0 "linear" 0 "open" 0 "4" 0>
  <.SP SP1 1 210 470 0 51 0 0 "lin" 1 "0 MHz" 1 "30 GHz" 1 "601" 1 "no" 0 "1" 0 "2" 0 "no" 0 "no" 0>
</Components>
<Wires>
  <200 240 200 260 "" 0 0 0 "">
  <200 240 360 240 "" 0 0 0 "">
  <200 320 200 340 "" 0 0 0 "">
  <280 300 280 320 "" 0 0 0 "">
  <280 300 360 300 "" 0 0 0 "">
  <280 380 280 400 "" 0 0 0 "">
  <560 240 560 260 "" 0 0 0 "">
  <420 240 560 240 "" 0 0 0 "">
  <460 300 460 320 "" 0 0 0 "">
  <420 300 460 300 "" 0 0 0 "">
  <460 380 460 400 "" 0 0 0 "">
  <560 320 560 340 "" 0 0 0 "">
  <390 330 390 350 "" 0 0 0 "">
</Wires>
<Diagrams>
</Diagrams>
<Paintings>
</Paintings>
