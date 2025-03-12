const { exec } = require('child_process');

     module.exports = (req, res) => {
       exec('python decartes.py', (error, stdout, stderr) => {
         if (error) {
           res.status(500).send(`Erro: ${stderr}`);
           return;
         }
         res.send(stdout);
       });
     };
