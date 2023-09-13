while [ $(squeue -u michels9 | wc -l) -gt 6 ]; do
  squeue -u michels9
  sleep 10;
done;


